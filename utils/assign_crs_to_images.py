from pathlib import Path
from osgeo import gdal
from tqdm import tqdm
import multiprocessing
from functools import partial
import argparse


def process_image(file_path, output_dir, target_crs):
    """Worker function to process a single image."""
    output_file = output_dir / f"{file_path.stem}.tif"

    try:
        # Use gdal_translate to convert jpg to tif and set crs
        creation_options = [
            "COMPRESS=JPEG",
            "JPEG_QUALITY=85",
            "PHOTOMETRIC=YCBCR",
            "TILED=YES",
            "BIGTIFF=IF_SAFER",
        ]
        gdal.Translate(
            destName=str(output_file),
            srcDS=str(file_path),
            outputSRS=target_crs,
            creationOptions=creation_options,
        )
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")


def parse_arguments(args=None):
    """Set up and parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Assign CRS to images and convert to TIFF."
    )
    parser.add_argument(
        "--img-dir",
        type=Path,
        required=True,
        help="Path to the folder containing images",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="tiff_with_crs",
        help="Name of the output subfolder (default: tiff_with_crs)",
    )
    parser.add_argument(
        "--target-crs",
        default="EPSG:27700",
        help="Target CRS (default: EPSG:27700)",
    )
    parser.add_argument(
        "--singleprocessor",
        action="store_true",
        help="Use single process instead of multiprocessing (slower)",
    )
    return parser.parse_args(args)


def main(args):
    """Main orchestration function."""
    img_dir = args.img_dir.resolve()

    # Raise an error if the folder does not exist
    if not img_dir.exists():
        raise ValueError(f"Folder not found: {img_dir}")

    image_files = list(img_dir.glob("*.jpg"))

    # Raise an error if the folder is empty of JPGs
    if not image_files:
        raise ValueError(f"No JPG images found to process in {img_dir}")

    output_dir = img_dir / args.output_subdir
    output_dir.mkdir(exist_ok=True)

    target_crs = args.target_crs
    print(f"Found {len(image_files)} JPG files to process in {img_dir}")

    process_func = partial(process_image, output_dir=output_dir, target_crs=target_crs)

    if not args.singleprocessor:
        # Use available cores - 1
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        print(f"Using {num_workers} workers for processing.")

        with multiprocessing.Pool(num_workers) as pool:
            list(
                tqdm(
                    pool.imap_unordered(process_func, image_files),
                    total=len(image_files),
                    desc="Assigning CRS and converting to TIFF",
                )
            )
    else:
        print("Using single process.")
        for file_path in tqdm(image_files, desc="Assigning CRS and converting to TIFF"):
            process_func(file_path)

    print("\nProcessing complete.")


if __name__ == "__main__":
    parsed_args = parse_arguments()
    main(parsed_args)
