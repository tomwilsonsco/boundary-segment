import argparse
from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import GeometryCollection
from shapely.ops import unary_union
from shapely.strtree import STRtree


def split_by_local_union(source_geoms, mask_buffers, crs):
    """
    For each geometry in source_geoms, query an STRtree built from
    mask_buffers to find only the nearby buffers, union just those
    locally, then split the source geometry into inside/outside portions.

    This avoids creating one massive global union, which is the main
    bottleneck when lines cover large extents.
    """
    mask_list = list(mask_buffers)
    tree = STRtree(mask_list)

    inside_list = []
    outside_list = []

    for geom in source_geoms:
        if geom is None or geom.is_empty:
            inside_list.append(GeometryCollection())
            outside_list.append(GeometryCollection())
            continue
        idxs = tree.query(geom, predicate="intersects")
        if len(idxs):
            local_mask = unary_union([mask_list[i] for i in idxs])
            inside_list.append(geom.intersection(local_mask))
            outside_list.append(geom.difference(local_mask))
        else:
            inside_list.append(GeometryCollection())
            outside_list.append(geom)

    return (
        gpd.GeoSeries(inside_list, crs=crs),
        gpd.GeoSeries(outside_list, crs=crs),
    )


def filter_lines(geoseries, crs, label):
    """
    Filters a GeoSeries to keep only lines, returning a labelled GeoDataFrame.
    This replaces the slow python 'for' loop from extract_lines.
    """
    valid_geoms = geoseries[~geoseries.is_empty]

    lines_only = valid_geoms[
        valid_geoms.geom_type.isin(["LineString", "MultiLineString"])
    ]

    gdf = gpd.GeoDataFrame(geometry=lines_only, crs=crs)
    gdf["pred_result"] = label

    return gdf


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
        "--buffer-dist",
        type=float,
        default=3.0,
        help="Buffer distance for line evaluation in CRS units (e.g. metres). Default: 3.0.",
    )

    return parser.parse_args(args)


def main(args):
    """Main function to evaluate prediction lines."""

    print(f"Loading predictions from {args.pred_gpkg}...")
    pred_gdf = gpd.read_file(args.pred_gpkg)

    print(f"Loading parcels from {args.parcels}...")
    parcels_gdf = gpd.read_file(args.parcels)

    crs = pred_gdf.crs

    print("Converting parcels to boundary lines and buffering...")
    parcel_lines = parcels_gdf.geometry.boundary
    parcel_buffers = parcel_lines.buffer(args.buffer_dist)

    print("Splitting prediction lines (True Positives / False Positives)...")
    tp_geoms, fp_geoms = split_by_local_union(pred_gdf.geometry, parcel_buffers, crs)
    tp_gdf = filter_lines(tp_geoms, crs, "TP")
    fp_gdf = filter_lines(fp_geoms, crs, "FP")

    # FN
    print(f"Buffering prediction lines by {args.buffer_dist}...")
    pred_buffers = pred_gdf.geometry.buffer(args.buffer_dist)
    print("Evaluating ground truth lines for False Negatives (FN)...")
    _, fn_geoms = split_by_local_union(parcel_lines, pred_buffers, crs)
    fn_gdf = filter_lines(fn_geoms, crs, "FN")

    print("Combining results...")
    combined_df = pd.concat([tp_gdf, fp_gdf, fn_gdf], ignore_index=True)
    result_gdf = gpd.GeoDataFrame(combined_df, geometry="geometry", crs=crs)

    print("Dissolving geometries by prediction result...")
    result_gdf = result_gdf.dissolve(by="pred_result").reset_index()

    print("Exploding multi-part geometries...")
    result_gdf = result_gdf.explode(index_parts=True).reset_index(drop=True)

    output_gpkg = args.pred_gpkg.parent / f"{args.pred_gpkg.stem}_result_compare.gpkg"
    print(f"Saving evaluated lines to {output_gpkg}...")
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    result_gdf.to_file(output_gpkg, driver="GPKG")

    tp_len = tp_gdf.geometry.length.sum()
    fp_len = fp_gdf.geometry.length.sum()
    fn_len = fn_gdf.geometry.length.sum()

    print("\n" + "=" * 40)
    print("Evaluation Complete - Total Lengths")
    print(f"  TP length: {tp_len:,.1f} m")
    print(f"  FP length: {fp_len:,.1f} m")
    print(f"  FN length: {fn_len:,.1f} m")
    print("=" * 40)


if __name__ == "__main__":
    parsed_args = parse_arguments()
    main(parsed_args)
