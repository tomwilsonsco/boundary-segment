"""Chip a VRT mosaic into fixed-size GeoTIFF tiles with optional resampling."""

import argparse
import logging
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rschip import ImageChip
from shapely.geometry import box

# rschip names each chip <name>_<x>_<y>.tif, x and y being the top-left pixel of
# its read window in the source image
CHIP_NAME = re.compile(r"_(\d+)_(\d+)\.tif$")

# file operations on a mounted Windows drive spend their time waiting, so
# deleting chips on several threads is much quicker than one at a time
DELETE_THREADS = 16


def chip_index(chip_files, vrt_transform, read_dimensions, crs):
    """Chip footprints worked out from the chip names, without opening the chips.

    Each chip covers a read window of read_dimensions pixels square in the VRT,
    whatever the resampling, so its bounds follow from the window position.
    Opening every chip to read its bounds is very slow on a mounted drive.
    """
    geoms = []
    for file_path in chip_files:
        match = CHIP_NAME.search(file_path.name)
        if match is None:
            raise ValueError(
                f"Chip name {file_path.name} does not end in _<x>_<y>.tif; "
                "rschip's naming may have changed"
            )
        x, y = int(match[1]), int(match[2])
        window = Window(x, y, read_dimensions, read_dimensions)
        geoms.append(box(*window_bounds(window, vrt_transform)))
    return gpd.GeoDataFrame(
        {"file_name": [f.name for f in chip_files]}, geometry=geoms, crs=crs
    )


def check_chip_index(gdf, chip_files, sample_size=5):
    """Compare a sample of computed footprints with the chips' own bounds."""
    sample = np.unique(np.linspace(0, len(chip_files) - 1, sample_size).astype(int))
    for i in sample:
        with rio.open(chip_files[i]) as src:
            actual = np.array(src.bounds)
        computed = np.array(gdf.geometry.iloc[i].bounds)
        if not np.allclose(actual, computed, atol=0.01):
            raise ValueError(
                f"Computed bounds {computed} for {chip_files[i].name} do not match "
                f"its georeferencing {actual}; rschip's naming may have changed"
            )


def delete_files(paths):
    """Delete files on several threads, ignoring any already gone."""
    with ThreadPoolExecutor(max_workers=DELETE_THREADS) as pool:
        list(pool.map(lambda p: p.unlink(missing_ok=True), paths))


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
    parser.add_argument(
        "--prediction-mask",
        type=Path,
        default=None,
        help="Path to the prediction mask file (.gpkg or .shp). Chips that do not intersect "
        "the mask are deleted and recorded in chips_ignore.csv with condition "
        "'outside prediction mask'.",
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

    chip_files = sorted(out_dir.glob("*.tif"))
    if chip_files:
        with rio.open(vrt_path) as vrt_src:
            vrt_geom = box(*vrt_src.bounds)
            vrt_crs = vrt_src.crs
            vrt_transform = vrt_src.transform
        logging.info(f"Building index layer for {len(chip_files)} chips...")
        gdf = chip_index(
            chip_files, vrt_transform, image_chipper.read_dimensions, vrt_crs
        )
        check_chip_index(gdf, chip_files)

        # Load prediction mask if provided
        mask_geom = None
        if args.prediction_mask is not None:
            mask_path = args.prediction_mask.resolve()
            if not mask_path.exists():
                raise ValueError(f"Prediction mask not found: {mask_path}")
            mask_gdf = gpd.read_file(mask_path)
            if mask_gdf.empty:
                raise ValueError(f"Prediction mask contains no features: {mask_path}")
            mask_gdf = mask_gdf.to_crs(vrt_crs)
            mask_geom = mask_gdf.union_all()
            if mask_geom is None or mask_geom.is_empty:
                raise ValueError(
                    f"Prediction mask geometry is empty after union: {mask_path}"
                )

        # Collect all ignore records
        all_ignore_records = []

        # Find polygons not entirely within the VRT extent (small buffer for floating point precision)
        outside_mask = ~gdf.geometry.within(vrt_geom.buffer(0.001))
        outside_chips = gdf[outside_mask]
        if not outside_chips.empty:
            all_ignore_records.extend(
                {
                    "file_name": row["file_name"],
                    "remove_condition": "outside image bounds",
                }
                for _, row in outside_chips.iterrows()
            )

        # Filter by prediction mask (deleting non-intersecting chips from disk)
        if mask_geom is not None:
            # chips that exist on disk but do NOT intersect the mask
            mask_filter = gdf.geometry.intersects(mask_geom)
            chips_to_remove = gdf[~mask_filter]
            logging.info(
                f"Deleting {len(chips_to_remove)} chips outside the prediction mask..."
            )
            delete_files([out_dir / name for name in chips_to_remove["file_name"]])
            all_ignore_records.extend(
                {"file_name": name, "remove_condition": "outside prediction mask"}
                for name in chips_to_remove["file_name"]
            )
            # Keep only chips that pass both filters for the index
            keep_mask = mask_filter & ~outside_mask
            gdf = gdf[keep_mask].copy()
        else:
            gdf = gdf[~outside_mask].copy()

        # Write filtered index
        if not gdf.empty:
            gdf.to_file(out_dir / "chips_index.gpkg")
        else:
            logging.warning("No chips remain after filtering; skipping index write.")

        # Write chips_ignore.csv (if any records exist)
        if all_ignore_records:
            ignore_df = pd.DataFrame(all_ignore_records)
            ignore_df.to_csv(out_dir / "chips_ignore.csv", index=False)
            logging.info(
                f"Excluded {len(all_ignore_records)} chips: "
                f"{sum(1 for r in all_ignore_records if r['remove_condition'] == 'outside image bounds')} outside image bounds, "
                f"{sum(1 for r in all_ignore_records if r['remove_condition'] == 'outside prediction mask')} outside prediction mask."
            )
        else:
            logging.info("All chips are within the valid extent and mask.")


if __name__ == "__main__":
    parsed_args = parse_arguments()
    main(parsed_args)
