"""Chip a VRT mosaic into fixed-size GeoTIFF tiles with optional resampling."""

import logging
from rschip import ImageChip
from pathlib import Path
from tqdm import tqdm
import rasterio as rio
from shapely.geometry import box
import geopandas as gpd
import pandas as pd
import shutil
import argparse
import sys
import json


def parse_arguments(args=None):
    """Set up and parse command line arguments."""
    parser = argparse.ArgumentParser(description="Chip VRT image into smaller tiles.")
    parser.add_argument(
        "--vrt",
        type=Path,
        required=True,
        help="Path to the input VRT file",
    )
    parser.add_argument(
        "--output-subdir",
        default="chips",
        help="Name of the output subfolder (default: chips)",
    )
    parser.add_argument(
        "--chip-size",
        type=int,
        default=512,
        help="Height and width of each chip image window in pixels",
    )
    parser.add_argument(
        "--chip-offset",
        type=int,
        default=384,
        help="Step size (in pixels) between the start of adjacent chips. Must be less than --chip-size. "
        "An overlap equal to (chip-size minus chip-offset) ensures boundary features are not clipped at the edge of every chip. Default: 384.",
    )
    parser.add_argument(
        "--resampling-factor",
        type=float,
        default=0.5,
        help="Scale factor applied to chips at creation time. A value of 1.0 produces chips at the VRT resolution. "
        "A value of 0.5 halves the resolution (e.g. 0.125 m/px / 0.25 m/px). Default: 0.5",
    )
    parser.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help="If set, delete and recreate the output directory if it already exists. "
        "If not set, the script will exit if the output directory is non-empty.",
    )
    return parser.parse_args(args)


def main(args):
    """Main orchestration function."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    vrt_path = args.vrt.resolve()

    if not vrt_path.exists():
        raise ValueError(f"VRT file not found: {vrt_path}")

    if args.chip_offset >= args.chip_size:
        raise ValueError(
            f"Offset ({args.chip_offset}) must be smaller than chip size ({args.chip_size})"
        )

    out_dir = vrt_path.parent / args.output_subdir
    out_dir.mkdir(exist_ok=True)

    # if output directory is not empty, prompt user to overwrite
    if any(out_dir.iterdir()):
        if args.overwrite_output_dir:
            logging.info("Deleting existing files...")
            for file in out_dir.iterdir():
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    shutil.rmtree(file)
        else:
            logging.warning("Operation cancelled.")
            sys.exit(1)

    # initialize rschip.ImageChip
    image_chipper = ImageChip(
        input_image_path=vrt_path,
        output_path=out_dir,
        pixel_dimensions=args.chip_size,
        offset=args.chip_offset,
        scale_factor=args.resampling_factor,
    )

    # generate chips
    image_chipper.chip_image()

    geoms = []
    for file_path in tqdm(list(out_dir.glob("*.tif")), desc="building index layer"):
        with rio.open(file_path) as src:
            bounds = src.bounds
            geom = box(*bounds)
        geoms.append({"geometry": geom, "file_name": file_path.name})

    if geoms:
        with rio.open(list(out_dir.glob("*.tif"))[0]) as src:
            crs = src.crs
        gdf = gpd.GeoDataFrame(geoms, crs=crs)
        gdf.to_file(out_dir / "chips_index.gpkg")

        with rio.open(vrt_path) as vrt_src:
            vrt_geom = box(*vrt_src.bounds)

        # Find polygons not entirely within the VRT extent (small buffer for floating point precision)
        outside_mask = ~gdf.geometry.within(vrt_geom.buffer(0.001))
        outside_chips = gdf[outside_mask]
        if not outside_chips.empty:
            ignore_df = pd.DataFrame(
                {
                    "file_name": outside_chips["file_name"],
                    "remove_condition": "outside image bounds",
                }
            )
            ignore_df.to_csv(out_dir / "chips_ignore.csv", index=False)

    if args.sample_scaler:
        scaler = image_chipper.sample_to_scaler(int(1e5))
        with open(out_dir / "scaler.json", "w") as f:
            json.dump(scaler, f, indent=4)


if __name__ == "__main__":
    parsed_args = parse_arguments()
    main(parsed_args)
