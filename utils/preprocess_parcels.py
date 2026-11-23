"""Prepare an area's land parcels and prediction mask from the national parcels dataset.

The aerial photography extent is the union of the raw JPEG footprints, read from
their world files. Parcels intersecting the extent are selected from the national
dataset and clipped to it. The prediction mask is built from the unclipped
parcels, so the area threshold uses each parcel's full area: parcels smaller than
the threshold are dissolved, buffered and then clipped to the extent.
"""

import argparse
import logging
from pathlib import Path

import geopandas as gpd
from osgeo import gdal
from shapely.geometry import box
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

gdal.UseExceptions()


def image_footprint(file_path):
    """Footprint of a JPEG from its world file (.jgw), as a shapely box."""
    ds = gdal.Open(str(file_path))
    gt = ds.GetGeoTransform(can_return_null=True)
    width, height = ds.RasterXSize, ds.RasterYSize
    ds = None
    if gt is None:
        raise ValueError(f"No world file or georeferencing found for {file_path}")
    if gt[2] != 0 or gt[4] != 0:
        raise ValueError(f"Rotated georeferencing is not supported: {file_path}")
    x0, y0 = gt[0], gt[3]
    x1, y1 = x0 + width * gt[1], y0 + height * gt[5]
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def aerial_extent(img_dir, crs):
    """Dissolved footprint of all JPEGs in img_dir, as a one-row GeoDataFrame."""
    image_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.JPG")))
    if not image_files:
        raise ValueError(f"No JPG images found in {img_dir}")
    logging.info(f"Reading footprints of {len(image_files)} JPG files in {img_dir}")
    footprints = gpd.GeoSeries(
        [image_footprint(f) for f in tqdm(image_files, desc="Reading footprints")],
        crs=crs,
    )
    extent = gpd.GeoDataFrame(geometry=[footprints.union_all()], crs=crs)
    logging.info(
        f"Aerial extent: {extent.area.iloc[0] / 1e6:.1f} km2, "
        f"bounds {[round(v) for v in extent.total_bounds]}"
    )
    return extent


def load_intersecting_parcels(national_parcels, extent, crs):
    """Parcels from the national dataset that intersect the extent, unclipped."""
    logging.info(f"Reading parcels within the extent bounds from {national_parcels}")
    # the bbox filter uses the shapefile's spatial index, so only nearby
    # parcels are read from the national layer
    parcels = gpd.read_file(national_parcels, bbox=tuple(extent.total_bounds))
    if parcels.crs is None:
        parcels = parcels.set_crs(crs)
    elif parcels.crs != extent.crs:
        parcels = parcels.to_crs(crs)

    invalid = ~parcels.is_valid
    if invalid.any():
        logging.info(f"Repairing {invalid.sum()} invalid parcel geometries")
        parcels.loc[invalid, "geometry"] = parcels.loc[invalid].make_valid()

    extent_geom = extent.geometry.iloc[0]
    idx = parcels.sindex.query(extent_geom, predicate="intersects")
    parcels = parcels.iloc[sorted(idx)].reset_index(drop=True)
    if parcels.empty:
        raise ValueError("No parcels in the national dataset intersect the extent")
    logging.info(f"Selected {len(parcels)} parcels intersecting the aerial extent")
    return parcels


def build_prediction_mask(parcels, extent, threshold_ha, buffer_dist):
    """Dissolve parcels under the area threshold, buffer, and clip to the extent."""
    small = parcels[parcels.area < threshold_ha * 10_000]
    logging.info(
        f"Prediction mask: {len(small)} of {len(parcels)} parcels are smaller than "
        f"{threshold_ha} ha (measured before clipping)"
    )
    if small.empty:
        raise ValueError(f"No parcels smaller than {threshold_ha} ha in the extent")
    mask_geom = small.union_all().buffer(buffer_dist)
    mask_geom = mask_geom.intersection(extent.geometry.iloc[0])
    return gpd.GeoDataFrame(geometry=[mask_geom], crs=parcels.crs)


def write_layer(gdf, path):
    """Write a GeoDataFrame to a GeoPackage layer named after the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, layer=path.stem, driver="GPKG")
    logging.info(f"Wrote {len(gdf)} features to {path}")


def parse_arguments(args=None):
    """Set up and parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create an area's parcels and prediction mask from the national "
        "parcels dataset, clipped to the extent of the raw aerial JPEGs."
    )
    parser.add_argument(
        "--img-dir",
        type=Path,
        required=True,
        help="Folder of raw APGB JPEGs with world files (.jgw)",
    )
    parser.add_argument(
        "--national-parcels",
        type=Path,
        required=True,
        help="National land parcels dataset (.shp or .gpkg)",
    )
    parser.add_argument(
        "--output-parcels",
        type=Path,
        default=None,
        help="Output GeoPackage of parcels clipped to the aerial extent. If not "
        "given, no parcels file is written.",
    )
    parser.add_argument(
        "--output-mask",
        type=Path,
        default=None,
        help="Output GeoPackage for the prediction mask. If not given, no mask is made.",
    )
    parser.add_argument(
        "--output-extent",
        type=Path,
        default=None,
        help="Optional output GeoPackage of the aerial photography extent",
    )
    parser.add_argument(
        "--mask-threshold-ha",
        type=float,
        default=50.0,
        help="Parcels smaller than this (hectares, before clipping) form the "
        "prediction mask. Default: 50.",
    )
    parser.add_argument(
        "--mask-buffer",
        type=float,
        default=50.0,
        help="Buffer (metres) applied to the dissolved mask parcels. Default: 50.",
    )
    parser.add_argument(
        "--crs",
        default="EPSG:27700",
        help="CRS of the JPEG world files and the outputs. Default: EPSG:27700 "
        "(British National Grid).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist",
    )
    return parser.parse_args(args)


def main(args):
    """Build the extent, clipped parcels and prediction mask for one area."""
    img_dir = args.img_dir.resolve()
    if not img_dir.exists():
        raise ValueError(f"Folder not found: {img_dir}")
    if not args.national_parcels.exists():
        raise ValueError(f"National parcels dataset not found: {args.national_parcels}")
    if args.output_parcels is None and args.output_mask is None:
        raise ValueError("Give --output-parcels, --output-mask or both")

    # the extent is cheap to rebuild, but parcels and masks are not overwritten
    # without asking, as earlier areas' files were made by hand
    outputs = [args.output_parcels, args.output_mask]
    existing = [p for p in outputs if p is not None and p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {', '.join(map(str, existing))}. "
            "Use --overwrite to replace existing outputs."
        )

    extent = aerial_extent(img_dir, args.crs)
    if args.output_extent is not None:
        write_layer(extent, args.output_extent)

    parcels = load_intersecting_parcels(args.national_parcels, extent, args.crs)

    if args.output_parcels is not None:
        # clipping only the intersecting subset is much quicker than clipping the
        # national layer
        clipped = gpd.clip(parcels, extent, keep_geom_type=True)
        clipped = clipped[~clipped.is_empty].reset_index(drop=True)
        write_layer(clipped, args.output_parcels)

    if args.output_mask is not None:
        mask = build_prediction_mask(
            parcels, extent, args.mask_threshold_ha, args.mask_buffer
        )
        write_layer(mask, args.output_mask)


if __name__ == "__main__":
    parsed_args = parse_arguments()
    main(parsed_args)
