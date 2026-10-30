"""Compare two prediction runs: line-length metrics, line geometry and chip-seam statistics, with a spatial block bootstrap of the metric differences."""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import shapely

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_arguments(args=None):
    """Set up and parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare a candidate prediction run against a baseline run. "
        "Both runs must have been evaluated with line_evaluate.py. Its TP/FP/FN "
        "lines are read from <prediction>_result_compare.gpkg unless "
        "--baseline-eval / --candidate-eval give the path."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Baseline prediction lines GPKG (output of predict.py).",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Candidate prediction lines GPKG (output of predict.py).",
    )
    parser.add_argument(
        "--baseline-eval",
        type=Path,
        default=None,
        help="TP/FP/FN lines GPKG from line_evaluate.py for the baseline. Needed for "
        "run_pipeline.py runs, which name evaluations by buffer: use "
        "<run_id>_base_result_compare.gpkg or <run_id>_<desc>_result_compare.gpkg "
        "(report buffer), not <run_id>_base_filter_result_compare.gpkg. Default: "
        "<baseline stem>_result_compare.gpkg next to the baseline.",
    )
    parser.add_argument(
        "--candidate-eval",
        type=Path,
        default=None,
        help="As --baseline-eval, for the candidate. Compare evaluations made at the "
        "same buffer distance.",
    )
    parser.add_argument(
        "--chip-dir",
        type=Path,
        required=True,
        help="Directory with the chip TIFFs and chips_index.gpkg, used to find the "
        "chip grid that seam statistics are measured against.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Skeletonisation chunk size in pixels used by predict.py, to measure "
        "line endpoints at chunk edges as well as chip edges. Default: 2048.",
    )
    parser.add_argument(
        "--block-size",
        type=float,
        default=1000.0,
        help="Side of the square blocks for the spatial bootstrap, in CRS units. "
        "Default: 1000.",
    )
    parser.add_argument(
        "--min-block-length",
        type=float,
        default=2000.0,
        help="Blocks with less ground-truth boundary than this (TP + FN length) are "
        "left out of the bootstrap. Default: 2000.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=2000,
        help="Number of bootstrap resamples. Default: 2000.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save all results as JSON.",
    )
    return parser.parse_args(args)


def result_compare_path(pred_gpkg):
    return pred_gpkg.parent / f"{pred_gpkg.stem}_result_compare.gpkg"


def chip_grid(chip_dir):
    """Return the chip grid origin (left, top), chip step and pixel size."""
    index_path = chip_dir / "chips_index.gpkg"
    if not index_path.exists():
        raise FileNotFoundError(f"Chips index not found: {index_path}")
    index_gdf = gpd.read_file(index_path)
    if index_gdf.empty:
        raise ValueError(f"Chips index is empty: {index_path}")
    left, _, _, top = index_gdf.total_bounds
    min_xs = np.unique(np.round(index_gdf.geometry.bounds["minx"].to_numpy(), 3))
    steps = np.diff(min_xs)
    if not len(steps):
        raise ValueError("Need at least two chip columns to find the chip step.")
    chip_step = float(steps.min())

    first_chip = chip_dir / index_gdf["file_name"].iloc[0]
    with rio.open(first_chip) as src:
        pixel_size = src.res[0]
    return left, top, chip_step, pixel_size


def line_metrics(compare_gdf):
    lengths = compare_gdf.geometry.length.groupby(compare_gdf["pred_result"]).sum()
    tp, fp, fn = (float(lengths.get(k, 0.0)) for k in ("TP", "FP", "FN"))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp_m": tp,
        "fp_m": fp,
        "fn_m": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def line_geometry_stats(pred_gdf, origin, periods):
    """
    Length, fragmentation and seam statistics for prediction lines.

    Endpoints are the degree-1 nodes of the dissolved, merged line network, so
    the measure does not depend on how a run splits or traces its lines. For each
    period, endpoint positions are binned (4 m bins) modulo the period, measured
    from the chip grid origin; the busiest bin over the median bin shows whether
    lines end at that spacing more often than chance (about 1.1 for no effect).
    """
    geoms = pred_gdf.geometry.values
    stored = float(shapely.length(geoms).sum())
    network = shapely.line_merge(shapely.union_all(geoms))
    parts = list(getattr(network, "geoms", [network]))
    dissolved = float(sum(p.length for p in parts))

    end_counts = Counter()
    for part in parts:
        if part.is_ring:
            continue
        end_counts[tuple(np.round(part.coords[0], 3))] += 1
        end_counts[tuple(np.round(part.coords[-1], 3))] += 1
    ends = np.array([pt for pt, n in end_counts.items() if n == 1]).reshape(-1, 2)

    stats = {
        "features": len(pred_gdf),
        "geom_types": pred_gdf.geom_type.value_counts().to_dict(),
        "stored_length_m": stored,
        "dissolved_length_m": dissolved,
        "stored_to_dissolved": stored / dissolved if dissolved else 0.0,
        "endpoints": int(len(ends)),
        "endpoints_per_km": len(ends) / (dissolved / 1000) if dissolved else 0.0,
        "seam_peaks": {},
    }
    left, top = origin
    for name, period in periods.items():
        peaks = []
        for offsets in (ends[:, 0] - left, top - ends[:, 1]):
            hist, _ = np.histogram(
                offsets % period, bins=max(int(round(period / 4)), 1), range=(0, period)
            )
            median = np.median(hist)
            peaks.append(float(hist.max() / median) if median > 0 else float("nan"))
        stats["seam_peaks"][name] = {"period_m": period, "x": peaks[0], "y": peaks[1]}
    return stats


def block_lengths(compare_gdf, block_size):
    """TP / FP / FN length per square block, assigned by each line's midpoint."""
    gdf = compare_gdf.explode(index_parts=False)
    mid = gdf.geometry.interpolate(0.5, normalized=True)
    table = pd.DataFrame(
        {
            "bx": np.floor(mid.x / block_size).astype(int),
            "by": np.floor(mid.y / block_size).astype(int),
            "label": gdf["pred_result"].to_numpy(),
            "length": gdf.geometry.length.to_numpy(),
        }
    )
    blocks = table.pivot_table(
        index=["bx", "by"],
        columns="label",
        values="length",
        aggfunc="sum",
        fill_value=0.0,
    )
    return blocks.reindex(columns=["TP", "FP", "FN"], fill_value=0.0)


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return np.array([precision, recall, f1])


def block_bootstrap(base_blocks, cand_blocks, min_length, n_boot, seed=0):
    """
    Resample blocks with replacement and recompute the pooled precision, recall
    and F1 difference (candidate minus baseline) each time.
    """
    blocks = base_blocks.join(cand_blocks.add_suffix("_c"), how="outer").fillna(0.0)
    blocks = blocks[(blocks["TP"] + blocks["FN"]) >= min_length]
    if blocks.empty:
        raise ValueError("No blocks have enough ground-truth boundary to compare.")

    base = blocks[["TP", "FP", "FN"]].to_numpy()
    cand = blocks[["TP_c", "FP_c", "FN_c"]].to_numpy()

    per_block_f1 = np.array(
        [_prf(*c)[2] - _prf(*b)[2] for b, c in zip(base, cand)], dtype=float
    )
    observed = _prf(*cand.sum(0)) - _prf(*base.sum(0))

    rng = np.random.default_rng(seed)
    diffs = np.empty((n_boot, 3))
    for i in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        diffs[i] = _prf(*cand[pick].sum(0)) - _prf(*base[pick].sum(0))

    result = {
        "blocks": int(len(blocks)),
        "f1_better_blocks": int((per_block_f1 > 1e-9).sum()),
        "f1_worse_blocks": int((per_block_f1 < -1e-9).sum()),
    }
    for i, name in enumerate(("precision", "recall", "f1")):
        lo, hi = np.percentile(diffs[:, i], [2.5, 97.5])
        result[name] = {
            "diff": float(observed[i]),
            "ci95": [float(lo), float(hi)],
            "p_diff_le_0": float(np.mean(diffs[:, i] <= 0)),
        }
    return result


def main(args):
    """Compare the two runs and log a summary."""
    eval_paths = {
        "baseline": args.baseline_eval or result_compare_path(args.baseline),
        "candidate": args.candidate_eval or result_compare_path(args.candidate),
    }
    for name, path in (("baseline", args.baseline), ("candidate", args.candidate)):
        if not path.exists():
            raise FileNotFoundError(f"Prediction GPKG not found: {path}")
        if not eval_paths[name].exists():
            raise FileNotFoundError(
                f"{eval_paths[name]} not found. Run line_evaluate.py on {path} first, "
                f"or pass --{name}-eval."
            )

    left, top, chip_step, pixel_size = chip_grid(args.chip_dir)
    periods = {
        "chip step": chip_step,
        "skeleton chunk": args.chunk_size * pixel_size,
        # spacings unrelated to the chip grid, showing the no-effect level
        "control A": round(chip_step * 0.87),
        "control B": round(chip_step * 1.05),
    }
    logging.info(f"Chip step {chip_step:g} m, pixel size {pixel_size:g} m.")

    results = {}
    blocks = {}
    for name, pred_path in (("baseline", args.baseline), ("candidate", args.candidate)):
        logging.info(f"Measuring {name}: {pred_path.name}...")
        compare_gdf = gpd.read_file(eval_paths[name])
        results[name] = {
            "prediction": str(pred_path),
            "evaluation": str(eval_paths[name]),
            "metrics": line_metrics(compare_gdf),
            "geometry": line_geometry_stats(
                gpd.read_file(pred_path), (left, top), periods
            ),
        }
        blocks[name] = block_lengths(compare_gdf, args.block_size)

    logging.info("Running spatial block bootstrap...")
    results["bootstrap"] = block_bootstrap(
        blocks["baseline"], blocks["candidate"], args.min_block_length, args.n_boot
    )

    base, cand = results["baseline"], results["candidate"]
    lines = ["", f"{'':44}{'baseline':>14}{'candidate':>14}"]
    for key in ("precision", "recall", "f1"):
        lines.append(
            f"{key:44}{base['metrics'][key]:>14.4f}{cand['metrics'][key]:>14.4f}"
        )
    for key, label in (("fp_m", "FP length (m)"), ("fn_m", "FN length (m)")):
        lines.append(
            f"{label:44}{base['metrics'][key]:>14,.0f}{cand['metrics'][key]:>14,.0f}"
        )
    for key, label in (
        ("stored_to_dissolved", "stored / dissolved length"),
        ("endpoints_per_km", "line endpoints per km"),
    ):
        lines.append(
            f"{label:44}{base['geometry'][key]:>14.2f}{cand['geometry'][key]:>14.2f}"
        )
    for name, period in periods.items():
        b = base["geometry"]["seam_peaks"][name]
        c = cand["geometry"]["seam_peaks"][name]
        label = f"endpoint peak, {name} ({period:g} m) x/y"
        lines.append(
            f"{label:44}{b['x']:>8.2f}/{b['y']:<5.2f}{c['x']:>8.2f}/{c['y']:<5.2f}"
        )
    boot = results["bootstrap"]
    lines.append("")
    lines.append(
        f"Block bootstrap ({boot['blocks']} blocks of {args.block_size:g} m, "
        f"{args.n_boot} resamples): F1 better in {boot['f1_better_blocks']}, "
        f"worse in {boot['f1_worse_blocks']}"
    )
    for key in ("precision", "recall", "f1"):
        lo, hi = boot[key]["ci95"]
        lines.append(
            f"  {key:10} {boot[key]['diff']:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
        )
    logging.info("\n".join(lines))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=1)
        logging.info(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    parsed_args = parse_arguments()
    main(parsed_args)
