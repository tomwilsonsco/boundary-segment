"""Rank parcels for review by how far their mapped boundaries disagree with the predicted lines.

Reads the TP/FP/FN lines written by line_evaluate.py, so the parcel signals add up
to the same lengths as the run's metrics, and a reviewer sees the lines behind each
score. One row per parcel, with two signals from the method review
(docs/method_review.md 3.1):
- missed edge: the length and share of the parcel boundary that is FN;
- interior lines: the length of FP line inside the parcel, more than --buffer-dist
  from its boundary.

A third, offset edge, is the part of the missed edge that has a predicted line (TP or
FP) within --offset-dist: a likely moved or mis-mapped boundary. It is reported, not
scored.

The score is max(missed_frac / --missed-frac-flag, interior_m / --interior-m-flag),
and a parcel is flagged when it is above 1. It is a ranking, not a probability: it has
not been calibrated against parcels known to have changed.
"""

import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

IN_SCOPE = "in scope"
OUTSIDE_MASK = "outside mask"
TOO_LARGE = "too large"

# metres; FN lines are pieces of the mapped boundaries, so they lie on the parcel
# edges up to floating point error
EDGE_TOLERANCE = 0.5


def parse_arguments(args=None):
    """Set up and parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Score and rank parcels for review from the TP/FP/FN lines of "
        "line_evaluate.py."
    )
    parser.add_argument(
        "--line-comparison",
        type=Path,
        required=True,
        help="TP/FP/FN lines GPKG from line_evaluate.py (<prediction>_result_compare"
        ".gpkg), made with the same parcels.",
    )
    parser.add_argument(
        "--parcels", type=Path, required=True, help="Land parcel polygons GPKG."
    )
    parser.add_argument(
        "--prediction-mask",
        type=Path,
        default=None,
        help="Optional prediction mask GPKG. Parcels with less than --min-mask-frac of "
        "their area inside it are kept in the output but not scored.",
    )
    parser.add_argument(
        "--output-gpkg",
        type=Path,
        default=None,
        help="Output GPKG. Default: the line comparison's name with _result_compare "
        "replaced by _parcel_review. An existing file is replaced.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to also write the parcel counts as JSON.",
    )
    parser.add_argument(
        "--buffer-dist",
        type=float,
        default=8.0,
        help="Metres; the buffer the line comparison was made with. FP lines within "
        "it of the parcel boundary are not counted as interior lines. Default: 8.0 "
        "(the business rule).",
    )
    parser.add_argument(
        "--offset-dist",
        type=float,
        default=30.0,
        help="Metres; a missed edge with a predicted line within this distance is "
        "counted as an offset edge. Default: 30.0.",
    )
    parser.add_argument(
        "--min-mask-frac",
        type=float,
        default=0.9,
        help="Share of a parcel's area that must be inside the prediction mask for it "
        "to be scored. Default: 0.9.",
    )
    parser.add_argument(
        "--max-parcel-area",
        type=float,
        default=5e5,
        help="Parcels larger than this (CRS units, m²) are not scored, as in "
        "line_evaluate.py. Default: 5e5 (50 ha).",
    )
    parser.add_argument(
        "--missed-frac-flag",
        type=float,
        default=0.25,
        help="Share of missed edge that on its own gives a score of 1. Default: 0.25.",
    )
    parser.add_argument(
        "--interior-m-flag",
        type=float,
        default=50.0,
        help="Metres of interior line that on their own give a score of 1. "
        "Default: 50.0.",
    )
    return parser.parse_args(args)


def mask_fractions(geoms, mask_geom):
    """Share of each geometry's area inside the mask.

    Parcels wholly inside the mask's interior are given 1 without intersecting
    them with it, which is slow for a detailed mask.
    """
    shapely.prepare(mask_geom)
    fracs = np.ones(len(geoms))
    partial = ~shapely.contains_properly(mask_geom, geoms)
    areas = shapely.area(geoms[partial])
    inside = shapely.area(shapely.intersection(geoms[partial], mask_geom))
    fracs[partial] = np.divide(
        inside, areas, out=np.zeros_like(inside), where=areas > 0
    )
    return fracs


class LineSets:
    """FN, FP and predicted (TP + FP) lines, each with a spatial index."""

    def __init__(self, comparison):
        result = comparison["pred_result"]
        geoms = comparison.geometry.values.to_numpy()
        self.fn = geoms[(result == "FN").to_numpy()]
        self.fp = geoms[(result == "FP").to_numpy()]
        self.pred = geoms[result.isin(["TP", "FP"]).to_numpy()]
        self.fn_tree = shapely.STRtree(self.fn)
        self.fp_tree = shapely.STRtree(self.fp)
        self.pred_tree = shapely.STRtree(self.pred)


def length_within(geom, tree, distance, step=1.0):
    """Length of a line geometry within distance of any geometry in tree.

    Measured from points every `step` metres along it, each standing for the
    length round it. Buffering the long merged prediction lines instead is far
    slower.
    """
    parts = shapely.get_parts(geom)
    parts = parts[shapely.length(parts) > 0]
    if len(parts) == 0:
        return 0.0
    points, weights = [], []
    for part in parts:
        n = max(1, int(np.ceil(part.length / step)))
        dists = (np.arange(n) + 0.5) * part.length / n
        points.append(shapely.line_interpolate_point(part, dists))
        weights.append(np.full(n, part.length / n))
    points = np.concatenate(points)
    weights = np.concatenate(weights)
    hit = np.unique(tree.query(points, predicate="dwithin", distance=distance)[0])
    return float(weights[hit].sum())


def parcel_signals(parcel, lines, buffer_dist, offset_dist):
    """Edge, missed edge, offset edge and interior line lengths for one parcel."""
    boundary = parcel.boundary
    edge_m = boundary.length

    missed = None
    missed_m = 0.0
    idx = lines.fn_tree.query(boundary, predicate="dwithin", distance=EDGE_TOLERANCE)
    if len(idx):
        on_edge = boundary.buffer(EDGE_TOLERANCE, quad_segs=2)
        missed = shapely.union_all(lines.fn[idx]).intersection(on_edge)
        missed_m = min(missed.length, edge_m)

    offset_m = 0.0
    if missed_m > 0:
        offset_m = min(length_within(missed, lines.pred_tree, offset_dist), missed_m)

    interior_m = 0.0
    core = parcel.buffer(-buffer_dist)
    if not core.is_empty:
        idx = lines.fp_tree.query(core, predicate="intersects")
        if len(idx):
            interior_m = shapely.union_all(lines.fp[idx]).intersection(core).length
    return edge_m, missed_m, offset_m, interior_m


def score_parcels(parcels, lines, args):
    """Add the scope, signal, score and rank columns to a copy of the parcels."""
    out = parcels.copy()
    geoms = out.geometry.values.to_numpy()
    area = shapely.area(geoms)
    out["area_ha"] = area / 1e4
    out["mask_frac"] = 1.0
    if args.mask_geom is not None:
        out["mask_frac"] = mask_fractions(geoms, args.mask_geom)
    out["scope"] = IN_SCOPE
    out.loc[out["mask_frac"] < args.min_mask_frac, "scope"] = OUTSIDE_MASK
    out.loc[(out["scope"] == IN_SCOPE) & (area > args.max_parcel_area), "scope"] = (
        TOO_LARGE
    )
    in_scope = (out["scope"] == IN_SCOPE).to_numpy()

    rows = [
        parcel_signals(g, lines, args.buffer_dist, args.offset_dist)
        for g in tqdm(geoms[in_scope], desc="Scoring parcels")
    ]
    signals = np.array(rows, dtype=float).reshape(-1, 4)
    for i, col in enumerate(["edge_m", "missed_edge_m", "offset_edge_m", "interior_m"]):
        out[col] = np.nan
        out.loc[in_scope, col] = signals[:, i]
    edge = out["edge_m"]
    out["missed_frac"] = (out["missed_edge_m"] / edge).where(edge > 0, 0.0)
    out.loc[~in_scope, "missed_frac"] = np.nan

    missed_part = out["missed_frac"] / args.missed_frac_flag
    interior_part = out["interior_m"] / args.interior_m_flag
    out["score"] = np.fmax(missed_part, interior_part)
    out["flag"] = out["score"] > 1.0
    out["main_signal"] = ""
    out.loc[in_scope, "main_signal"] = np.where(
        missed_part[in_scope] >= interior_part[in_scope],
        "missed edge",
        "interior lines",
    )
    out.loc[in_scope & (out["score"] == 0), "main_signal"] = ""

    # rank 1 is the parcel to review first; ties go to the larger disagreement
    order = (
        out[in_scope]
        .assign(_total=out["missed_edge_m"] + out["interior_m"])
        .sort_values(["score", "_total"], ascending=False)
        .index
    )
    out["rank"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out.loc[order, "rank"] = np.arange(1, len(order) + 1)

    out["review_status"] = ""
    out["review_notes"] = ""
    first = [
        "rank",
        "score",
        "flag",
        "main_signal",
        "missed_frac",
        "missed_edge_m",
        "offset_edge_m",
        "interior_m",
        "edge_m",
        "area_ha",
        "mask_frac",
        "scope",
        "review_status",
        "review_notes",
    ]
    rest = [c for c in parcels.columns if c != out.geometry.name]
    return out[rest + first + [out.geometry.name]]


def summarise(scored, args):
    """Parcel counts by scope and flag, for logging and JSON."""
    in_scope = scored[scored["scope"] == IN_SCOPE]
    n_flag = int(in_scope["flag"].sum())
    return {
        "line_comparison": str(args.line_comparison),
        "parcels": str(args.parcels),
        "prediction_mask": str(args.prediction_mask) if args.prediction_mask else None,
        "buffer_dist": args.buffer_dist,
        "offset_dist": args.offset_dist,
        "min_mask_frac": args.min_mask_frac,
        "max_parcel_area": args.max_parcel_area,
        "missed_frac_flag": args.missed_frac_flag,
        "interior_m_flag": args.interior_m_flag,
        "parcels_total": len(scored),
        "parcels_by_scope": scored["scope"].value_counts().to_dict(),
        "in_scope": len(in_scope),
        "flagged": n_flag,
        "flag_rate": n_flag / len(in_scope) if len(in_scope) else None,
        "over_missed_frac_flag": int(
            (in_scope["missed_frac"] > args.missed_frac_flag).sum()
        ),
        "over_interior_m_flag": int(
            (in_scope["interior_m"] > args.interior_m_flag).sum()
        ),
        "with_offset_edge": int((in_scope["offset_edge_m"] > 0).sum()),
    }


def main(args):
    """Score the parcels and write the ranked review layer."""
    for path in (args.line_comparison, args.parcels, args.prediction_mask):
        if path is not None and not path.exists():
            raise FileNotFoundError(f"Not found: {path}")

    logging.info(f"Loading parcels from {args.parcels}...")
    parcels = gpd.read_file(args.parcels)
    if parcels.empty:
        raise ValueError(f"{args.parcels} contains no parcels.")
    crs = parcels.crs
    if crs is None:
        raise ValueError(f"{args.parcels} has no CRS.")
    parcels = parcels[~parcels.geometry.is_empty & parcels.geometry.notna()]

    logging.info(f"Loading TP/FP/FN lines from {args.line_comparison}...")
    comparison = gpd.read_file(args.line_comparison)
    if comparison.crs is None:
        raise ValueError(f"{args.line_comparison} has no CRS.")
    if "pred_result" not in comparison.columns:
        raise ValueError(
            f"{args.line_comparison} has no 'pred_result' column; is it the output "
            "of line_evaluate.py?"
        )
    comparison = comparison.to_crs(crs)
    comparison = comparison[~comparison.geometry.is_empty & comparison.geometry.notna()]
    lines = LineSets(comparison)
    logging.info(
        f"{len(lines.fn)} FN, {len(lines.fp)} FP and {len(lines.pred)} predicted "
        "(TP + FP) lines."
    )

    args.mask_geom = None
    if args.prediction_mask is not None:
        mask = gpd.read_file(args.prediction_mask)
        if mask.empty:
            raise ValueError(f"{args.prediction_mask} contains no features.")
        args.mask_geom = mask.to_crs(crs).union_all()

    scored = score_parcels(parcels, lines, args)
    summary = summarise(scored, args)
    if summary["in_scope"] == 0:
        raise ValueError("No parcels are in scope, so none were scored.")

    stem = args.line_comparison.stem.removesuffix("_result_compare")
    out_gpkg = args.output_gpkg or (
        args.line_comparison.parent / f"{stem}_parcel_review.gpkg"
    )
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    if out_gpkg.exists():
        logging.info(f"Replacing {out_gpkg}")
        out_gpkg.unlink()
    scored.to_file(out_gpkg, layer="parcel_review", driver="GPKG")

    logging.info(
        f"{summary['in_scope']} of {summary['parcels_total']} parcels scored "
        f"({summary['parcels_by_scope']}). Flagged {summary['flagged']} "
        f"({summary['flag_rate']:.0%}): {summary['over_missed_frac_flag']} over "
        f"{args.missed_frac_flag:.0%} missed edge, "
        f"{summary['over_interior_m_flag']} over {args.interior_m_flag:g} m of "
        f"interior lines."
    )
    logging.info(f"Parcel review layer written to {out_gpkg}")
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2))
        logging.info(f"Summary written to {args.summary_json}")


if __name__ == "__main__":
    main(parse_arguments())
