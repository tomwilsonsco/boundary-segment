from pathlib import Path
from osgeo import gdal
import argparse


def parse_arguments(args=None):
    """Parse user arguments using argparse"""
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
        help="CRS to assign to the VRT (default: EPSG:27700)",
    )
    return parser.parse_args(args)


def main(args):
    """Main function."""
    img_dir = args.img_dir.resolve()

    if not img_dir.exists():
        raise ValueError(f"Folder not found: {img_dir}")

    image_files = list(img_dir.glob("*.tif")) + list(img_dir.glob("*.jpg"))

    if not image_files:
        raise ValueError(f"No TIFF or JPEG files found in {img_dir}")

    print(f"Found {len(image_files)} image files in {img_dir}")

    output_vrt = img_dir / args.output_filename
    print(f"Creating VRT at: {output_vrt}")

    target_crs = args.crs
    print(f"Assigning CRS: {target_crs}")

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
        print("VRT file created successfully!")

    except Exception as e:
        print(f"Error creating VRT: {e}")


if __name__ == "__main__":
    parsed_args = parse_arguments()
    main(parsed_args)
