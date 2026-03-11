from rschip import ImageChip
from pathlib import Path
from tqdm import tqdm
import rasterio as rio
from shapely.geometry import box
import geopandas as gpd
import shutil
import argparse
import sys


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
        help="How much offset between chips, for example if size 512 and"
        " offset of 384 this means an overlap of 128",
    )

    parser.add_argument(
        "--create-index-layer",
        action="store_true",
        help="Create a GPKG index layer of the chips",
    )
    parser.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help="Overwrite if output directory already exists. "
        "If not set the process will stop if output directory already exists.",
    )
    return parser.parse_args(args)


def main(args):
    """Main orchestration function."""
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
            print("Deleting existing files...")
            for file in out_dir.iterdir():
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    shutil.rmtree(file)
        else:
            print("Operation cancelled.")
            sys.exit()

    # initialize rschip.ImageChip
    image_chipper = ImageChip(
        input_image_path=vrt_path,
        output_path=out_dir,
        pixel_dimensions=args.chip_size,
        offset=args.chip_offset,
    )

    # generate chips
    image_chipper.chip_image()

    if args.create_index_layer:
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


if __name__ == "__main__":
    parsed_args = parse_arguments()
    main(parsed_args)
