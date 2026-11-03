"""Chip a VRT mosaic into fixed-size GeoTIFF tiles with optional resampling."""

import argparse
import logging
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rschip import ImageChip
from shapely import union_all
from shapely.geometry import box

# rschip names each chip <name>_<x>_<y>.tif, x and y being the top-left pixel of
# its read window in the source image
CHIP_NAME = re.compile(r"_(\d+)_(\d+)\.tif$")

# file operations on a mounted Windows drive spend their time waiting, so
# deleting chips on several threads is much quicker than one at a time
DELETE_THREADS = 16


def image_footprint(vrt_path, vrt_transform):
    """Area covered by the VRT's source images, from their positions in the VRT.

    The VRT's own bounds are the rectangle round all its sources. Where the tiles
    form an irregular block, that rectangle includes ground with no imagery.
    """
    band = ET.parse(vrt_path).getroot().find("VRTRasterBand")
    tiles = []
    for rect in band.iter("DstRect"):
        x, y, w, h = (float(rect.get(k)) for k in ("xOff", "yOff", "xSize", "ySize"))
        tiles.append(box(*window_bounds(Window(x, y, w, h), vrt_transform)))
    if not tiles:
        raise ValueError(f"No source images found in {vrt_path}")
    # the buffer closes floating point slivers between neighbouring tiles
    return union_all(tiles).buffer(0.01)


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
            vrt_crs = vrt_src.crs
            vrt_transform = vrt_src.transform
        vrt_geom = image_footprint(vrt_path, vrt_transform)
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

        # Chips not entirely within the imagery, so partly nodata
        outside_bounds = ~gdf.geometry.within(vrt_geom)
        if mask_geom is not None:
            in_mask = gdf.geometry.intersects(mask_geom)
        else:
            in_mask = pd.Series(True, index=gdf.index)

        # chips outside the mask are deleted, so each chip is recorded once:
        # under the mask if it was deleted, otherwise under the image bounds
        chips_to_remove = gdf[~in_mask]
        if not chips_to_remove.empty:
            logging.info(
                f"Deleting {len(chips_to_remove)} chips outside the prediction mask..."
            )
            delete_files([out_dir / name for name in chips_to_remove["file_name"]])
        ignore_df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "file_name": gdf.loc[in_mask & outside_bounds, "file_name"],
                        "remove_condition": "outside image bounds",
                    }
                ),
                pd.DataFrame(
                    {
                        "file_name": chips_to_remove["file_name"],
                        "remove_condition": "outside prediction mask",
                    }
                ),
            ],
            ignore_index=True,
        )
        gdf = gdf[in_mask & ~outside_bounds].copy()

        # Write filtered index
        if not gdf.empty:
            gdf.to_file(out_dir / "chips_index.gpkg")
        else:
            logging.warning("No chips remain after filtering; skipping index write.")

        # Write chips_ignore.csv (if any records exist)
        if not ignore_df.empty:
            ignore_df.to_csv(out_dir / "chips_ignore.csv", index=False)
            counts = ignore_df["remove_condition"].value_counts()
            logging.info(
                f"Excluded {len(ignore_df)} chips: "
                f"{counts.get('outside image bounds', 0)} outside image bounds, "
                f"{counts.get('outside prediction mask', 0)} outside prediction mask."
            )
        else:
            logging.info("All chips are within the valid extent and mask.")


if __name__ == "__main__":
    parsed_args = parse_arguments()
    main(parsed_args)
