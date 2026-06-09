"""Build a GDAL VRT mosaic from a directory of GeoTIFF or JPEG image files."""

import logging
from pathlib import Path
from osgeo import gdal
import argparse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_arguments(args=None):
    """Set up and parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a VRT mosaic from image files."
    )
    parser.add_argument(
        "--img-dir",
        type=Path,
        required=True,
        help="Path to the folder containing TIFF files",
    )
    parser.add_argument(
        "--output-filename",
        default="apgb_imgs.vrt",
        help="Name of the output VRT file (default: apgb_imgs.vrt)",
    )
    parser.add_argument(
        "--crs",
        type=str,
        default="EPSG:27700",
        help="CRS to assign to the VRT. Default: EPSG:27700 (British National Grid).",
    )
    return parser.parse_args(args)


def main(args):
    """Build a GDAL VRT mosaic from TIFF/JPEG files in the specified directory."""
    img_dir = args.img_dir.resolve()

    if not img_dir.exists():
        raise ValueError(f"Folder not found: {img_dir}")

    image_files = list(img_dir.glob("*.tif")) + list(img_dir.glob("*.jpg"))

    if not image_files:
        raise ValueError(f"No TIFF or JPEG files found in {img_dir}")

    logging.info(f"Found {len(image_files)} image files in {img_dir}")

    output_vrt = img_dir / args.output_filename
    logging.info(f"Creating VRT at: {output_vrt}")

    target_crs = args.crs
    logging.info(f"Assigning CRS: {target_crs}")

    try:
        options = gdal.BuildVRTOptions(
            resolution="highest",
            resampleAlg=gdal.GRA_NearestNeighbour,
            outputSRS=target_crs,
        )
        ds = gdal.BuildVRT(
            str(output_vrt), [str(f) for f in image_files], options=options
        )
        # Release the dataset so that the VRT is closed
        ds = None
        logging.info("VRT file created successfully!")

    except Exception as e:
        logging.error(f"Error creating VRT: {e}")


if __name__ == "__main__":
    parsed_args = parse_arguments()
    main(parsed_args)
