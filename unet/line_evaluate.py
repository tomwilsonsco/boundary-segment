import argparse
from datetime import datetime
from pathlib import Path
import geopandas as gpd
import pandas as pd
import rasterio as rio
from shapely.geometry import GeometryCollection, box
from shapely.ops import unary_union, substring, linemerge
from shapely.strtree import STRtree
from tqdm import tqdm


def extract_lines(geom):
    if geom is None or geom.is_empty:
        return GeometryCollection()
    if geom.geom_type in ["LineString", "MultiLineString"]:
        return geom
    if geom.geom_type == "GeometryCollection":
        lines = [
            g for g in geom.geoms if g.geom_type in ["LineString", "MultiLineString"]
        ]
        return unary_union(lines) if lines else GeometryCollection()
    return GeometryCollection()


def split_by_local_union(source_geoms, mask_buffers, crs):
    """
    For each geometry in source_geoms, query an STRtree built from
    mask_buffers. Find intersecting masks, union them, and compute
    intersection/difference for the line.
    """
    mask_list = list(mask_buffers)
    tree = STRtree(mask_list)

    inside_list = []
    outside_list = []

    for geom in tqdm(source_geoms, desc="Splitting by local union"):
        if geom is None or geom.is_empty:
            inside_list.append(GeometryCollection())
            outside_list.append(GeometryCollection())
            continue

        idxs = tree.query(geom, predicate="intersects")
        if len(idxs):
            local_mask = unary_union([mask_list[i] for i in idxs])
            inside_list.append(extract_lines(geom.intersection(local_mask)))
            outside_list.append(extract_lines(geom.difference(local_mask)))
        else:
            inside_list.append(GeometryCollection())
            outside_list.append(geom)

    return gpd.GeoSeries(inside_list, crs=crs), gpd.GeoSeries(outside_list, crs=crs)


def filter_lines(geoseries, crs, label):
    """
    Filters a GeoSeries to keep only lines, returning a labelled GeoDataFrame.
    """
    valid_geoms = geoseries[~geoseries.is_empty]

    lines_only = valid_geoms[
        valid_geoms.geom_type.isin(["LineString", "MultiLineString"])
    ]

    gdf = gpd.GeoDataFrame(geometry=lines_only, crs=crs)
    gdf["pred_result"] = label

    return gdf


def merge_lines(gdf):
    """
    Merge connected line segments within a classification GDF into the longest
    continuous features possible. Disconnected segments remain separate.
    """
    if gdf.empty:
        return gdf
    flat = []
    for geom in gdf.geometry:
        if geom.geom_type == "LineString":
            flat.append(geom)
        elif geom.geom_type == "MultiLineString":
            flat.extend(geom.geoms)
    merged = linemerge(flat)
    label = gdf["pred_result"].iloc[0]
    crs = gdf.crs
    if merged.geom_type == "LineString":
        geoms = [merged]
    else:
        geoms = list(merged.geoms)
    result = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    result["pred_result"] = label
    return result


def parse_arguments(args=None):
    """Set up and parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate predicted boundary lines against ground truth parcels."
    )

    parser.add_argument(
        "--pred-gpkg",
        type=Path,
        required=True,
        help="Path to the prediction lines GPKG.",
    )
    parser.add_argument(
        "--parcels",
        type=Path,
        required=True,
        help="Path to the ground truth parcels shapefile or GPKG.",
    )
    parser.add_argument(
        "--imgs-dir",
        type=Path,
        required=True,
        help="Directory of chip TIFFs to limit evaluation extent.",
    )
    parser.add_argument(
        "--buffer-dist",
        type=float,
        default=3.0,
        help="Buffer distance for line evaluation in CRS units (e.g. metres). Default: 3.0.",
    )
    parser.add_argument(
        "--max-parcel-area",
        type=float,
        default=5e5,
        help="Maximum parcel area in CRS units. Parcels larger than this are ignored. Default: 5e5.",
    )

    return parser.parse_args(args)


def main(args):
    """Main function to evaluate prediction lines."""

    print(f"Loading predictions from {args.pred_gpkg}...")
    pred_gdf = gpd.read_file(args.pred_gpkg)

    print(f"Loading parcels from {args.parcels}...")
    parcels_gdf = gpd.read_file(args.parcels)

    crs = pred_gdf.crs

    if parcels_gdf.crs != crs:
        parcels_gdf = parcels_gdf.to_crs(crs)

    print(f"Filtering parcels by area <= {args.max_parcel_area}...")
    parcels_gdf = parcels_gdf[parcels_gdf.geometry.area <= args.max_parcel_area]

    print("Clipping prediction lines by area filtered parcels...")
    if parcels_gdf.empty:
        pred_gdf = pred_gdf.iloc[0:0].copy()
    else:
        # buffer parcels by 10 specifically for the clipping process
        clip_parcel_list = list(parcels_gdf.geometry.buffer(10))
        tree = STRtree(clip_parcel_list)

        clipped_geoms = []
        keep_indices = []

        for i, geom in enumerate(tqdm(pred_gdf.geometry, desc="Clipping lines")):
            if geom is None or geom.is_empty:
                continue

            idxs = tree.query(geom, predicate="intersects")
            if len(idxs) > 0:
                local_mask = unary_union([clip_parcel_list[idx] for idx in idxs])
                clipped_geom = extract_lines(geom.intersection(local_mask))
                if not clipped_geom.is_empty:
                    clipped_geoms.append(clipped_geom)
                    keep_indices.append(i)

        pred_gdf = pred_gdf.iloc[keep_indices].copy()
        pred_gdf = pred_gdf.set_geometry(
            gpd.GeoSeries(clipped_geoms, index=pred_gdf.index, crs=crs)
        )
        pred_gdf = pred_gdf[
            pred_gdf.geom_type.isin(["LineString", "MultiLineString"])
        ].copy()

    if not args.imgs_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {args.imgs_dir}")

    index_path = args.imgs_dir / "chips_index.gpkg"

    if index_path.exists():
        print(f"Loading existing index layer from {index_path}...")
        index_gdf = gpd.read_file(index_path)
    else:
        print(f"Building index layer from {args.imgs_dir}...")
        tif_files = list(args.imgs_dir.glob("*.tif"))
        if not tif_files:
            raise FileNotFoundError(f"No .tif files found in {args.imgs_dir}")

        geoms = []
        for file_path in tqdm(tif_files, desc="building index layer"):
            with rio.open(file_path) as src:
                bounds = src.bounds
                geom = box(*bounds)
            geoms.append({"geometry": geom, "file_name": file_path.name})

        with rio.open(tif_files[0]) as src:
            index_crs = src.crs
        index_gdf = gpd.GeoDataFrame(geoms, crs=index_crs)
        print(f"Saving index layer to {index_path}...")
        index_gdf.to_file(index_path, driver="GPKG")

    if index_gdf.empty:
        raise ValueError("Chips index is empty. Cannot determine evaluation extent.")

    if index_gdf.crs != crs:
        index_gdf = index_gdf.to_crs(crs)

    index_union = index_gdf.union_all()
    print("Selecting parcels intersecting chips index...")
    intersecting_parcels = parcels_gdf[parcels_gdf.intersects(index_union)]

    print("Converting intersecting parcels to boundary lines...")
    parcel_lines = intersecting_parcels.geometry.boundary

    print("Clipping boundary lines by chips index...")
    parcel_lines_gdf = gpd.GeoDataFrame(geometry=parcel_lines, crs=crs)
    parcel_lines_gdf = gpd.clip(parcel_lines_gdf, index_union)

    print("Removing parcel lines near the external boundary of the study area...")
    extent_boundary_buffer = index_union.boundary.buffer(1.0)
    trimmed_geoms = parcel_lines_gdf.geometry.difference(extent_boundary_buffer)
    parcel_lines_gdf = parcel_lines_gdf.copy()
    parcel_lines_gdf["geometry"] = trimmed_geoms
    parcel_lines_gdf = parcel_lines_gdf[
        ~parcel_lines_gdf.geometry.is_empty
    ].reset_index(drop=True)

    parcel_lines = parcel_lines_gdf.geometry

    print("Clipping prediction lines by chips index...")
    pred_gdf = gpd.clip(pred_gdf, index_union)

    print("Buffering parcel lines...")
    parcel_buffers = parcel_lines.buffer(args.buffer_dist, resolution=4)

    print("Splitting prediction lines (True Positives / False Positives)...")
    tp_geoms, fp_geoms = split_by_local_union(pred_gdf.geometry, parcel_buffers, crs)
    tp_gdf = merge_lines(filter_lines(tp_geoms, crs, "TP"))
    fp_gdf = merge_lines(filter_lines(fp_geoms, crs, "FP"))

    # FN
    print("Exploding and simplifying prediction lines...")
    exploded_preds = pred_gdf.explode(index_parts=False)

    simplified_pred_lines = exploded_preds.geometry.simplify(1.0)

    print(f"Buffering prediction lines by {args.buffer_dist}...")
    pred_buffers = simplified_pred_lines.buffer(args.buffer_dist, resolution=4)
    print("Evaluating ground truth lines for False Negatives (FN)...")
    _, fn_geoms = split_by_local_union(parcel_lines, pred_buffers, crs)
    fn_gdf = merge_lines(filter_lines(fn_geoms, crs, "FN"))

    # Compute metrics from individual GDFs before any memory-intensive write
    tp_len = tp_gdf.geometry.length.sum()
    fp_len = fp_gdf.geometry.length.sum()
    fn_len = fn_gdf.geometry.length.sum()

    precision = tp_len / (tp_len + fp_len) if (tp_len + fp_len) > 0 else 0.0
    recall = tp_len / (tp_len + fn_len) if (tp_len + fn_len) > 0 else 0.0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    results_text = []
    results_text.append("=" * 60)
    results_text.append(f"LINE EVALUATION RESULTS - {timestamp}")
    results_text.append("=" * 60)
    results_text.append(f"Prediction: {args.pred_gpkg.name}")
    results_text.append(f"Ground Truth: {args.parcels.name}")
    results_text.append(f"Buffer distance: {args.buffer_dist} m")
    results_text.append("")
    results_text.append("Total Lengths:")
    results_text.append(f"  TP length: {tp_len:,.1f} m")
    results_text.append(f"  FP length: {fp_len:,.1f} m")
    results_text.append(f"  FN length: {fn_len:,.1f} m")
    results_text.append("")
    results_text.append("Metrics (based on line length):")
    results_text.append(f"  Precision: {precision:.4f}")
    results_text.append(f"  Recall:    {recall:.4f}")
    results_text.append(f"  F1 Score:  {f1_score:.4f}")
    results_text.append("=" * 60)
    results_text.append("")

    print("\n" + "\n".join(results_text))

    log_file = args.pred_gpkg.parent / "line_evaluate_results.log"
    with open(log_file, "a") as f:
        f.write("\n".join(results_text) + "\n")

    print(f"Results appended to {log_file}")

    # Write output in chunks, one component at a time, to avoid OOM from a
    # single large concat + to_file call.
    output_gpkg = args.pred_gpkg.parent / f"{args.pred_gpkg.stem}_result_compare.gpkg"
    print(f"Saving evaluated lines to {output_gpkg}...")
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    if output_gpkg.exists():
        output_gpkg.unlink()

    CHUNK_SIZE = 50_000
    first_write = True
    print("Combining results and writing in chunks...")
    for label, part_gdf in [("TP", tp_gdf), ("FP", fp_gdf), ("FN", fn_gdf)]:
        exploded = part_gdf.explode(index_parts=False).reset_index(drop=True)
        n = len(exploded)
        for i in range(0, n, CHUNK_SIZE):
            chunk = exploded.iloc[i : i + CHUNK_SIZE]
            if first_write:
                chunk.to_file(output_gpkg, driver="GPKG")
                first_write = False
            else:
                chunk.to_file(output_gpkg, driver="GPKG", mode="a")
        print(f"  {label}: wrote {n:,} features")
        del exploded

    print(f"Done. Output saved to {output_gpkg}")


if __name__ == "__main__":
    parsed_args = parse_arguments()
    main(parsed_args)
