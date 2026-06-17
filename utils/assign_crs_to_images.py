"""Assign a CRS to JPEG images and convert them to GeoTIFF format."""

import logging
from pathlib import Path
from osgeo import gdal
from tqdm import tqdm
import multiprocessing
from functools import partial
import argparse
import geopandas as gpd
import rasterio
from shapely.geometry import box
from shapely import wkt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def process_image(file_path, output_dir, target_crs, mask_wkt=None):
    """Worker function to process a single image.

    Returns:
        True if the file was written and kept,
        False if it was skipped (outside prediction mask),
        None if an error occurred.
    """
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
        logging.error(f"Error processing {file_path.name}: {e}")
        return None

    # If prediction mask is provided, check intersection
    if mask_wkt is not None:
        mask_geom = wkt.loads(mask_wkt)
        with rasterio.open(output_file) as src:
            bounds = src.bounds
        image_bbox = box(*bounds)
        if not image_bbox.intersects(mask_geom):
            output_file.unlink()
            logging.debug(f"{file_path.name} is outside prediction mask, deleted.")
            return False

    return True


def parse_arguments(args=None):
    """Set up and parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Assign CRS to images and convert to TIFF."
    )
    parser.add_argument(
        "--img-dir",
        type=Path,
        required=True,
        help="Path to the folder containing *.jpg / *.JPG images to process",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="tiff_with_crs",
        help="Name of the output subfolder (default: tiff_with_crs)",
    )
    parser.add_argument(
        "--crs",
        default="EPSG:27700",
        help="Target CRS to assign to output GeoTIFFs. Default: EPSG:27700 (British National Grid).",
    )
    parser.add_argument(
        "--singleprocessor",
        action="store_true",
        help="Use single process instead of multiprocessing (slower)",
    )
    parser.add_argument(
        "--prediction-mask",
        type=Path,
        default=None,
        help="Path to the prediction mask file (.gpkg or .shp). When provided, only GeoTIFFs whose "
        "bounding box intersects the mask are written.",
    )
    return parser.parse_args(args)


def main(args):
    """Main orchestration function."""
    img_dir = args.img_dir.resolve()

    # Raise an error if the folder does not exist
    if not img_dir.exists():
        raise ValueError(f"Folder not found: {img_dir}")

    image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.JPG"))

    # Raise an error if the folder is empty of JPGs
    if not image_files:
        raise ValueError(f"No JPG images found to process in {img_dir}")

    output_dir = img_dir / args.output_subdir
    output_dir.mkdir(exist_ok=True)

    crs = args.crs
    logging.info(f"Found {len(image_files)} JPG files to process in {img_dir}")

    # Load prediction mask if provided
    mask_wkt = None
    if args.prediction_mask is not None:
        mask_path = args.prediction_mask.resolve()
        if not mask_path.exists():
            raise ValueError(f"Prediction mask not found: {mask_path}")
        mask_gdf = gpd.read_file(mask_path)
        mask_gdf = mask_gdf.to_crs(crs)
        mask_union = mask_gdf.union_all()
        mask_wkt = mask_union.wkt
        logging.info(
            f"Loaded prediction mask from {mask_path}, reprojected to {crs}"
        )

    process_func = partial(
        process_image, output_dir=output_dir, target_crs=crs, mask_wkt=mask_wkt
    )

    written = 0
    skipped = 0
    errors = 0

    if not args.singleprocessor:
        # Use available cores - 1
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        logging.info(f"Using {num_workers} workers for processing.")

        with multiprocessing.Pool(num_workers) as pool:
            for result in tqdm(
                pool.imap_unordered(process_func, image_files),
                total=len(image_files),
                desc="Assigning CRS and converting to TIFF",
            ):
                if result is None:
                    errors += 1
                elif result is False:
                    skipped += 1
                else:
                    written += 1
    else:
        logging.info("Using single process.")
        for file_path in tqdm(
            image_files, desc="Assigning CRS and converting to TIFF"
        ):
            result = process_func(file_path)
            if result is None:
                errors += 1
            elif result is False:
                skipped += 1
            else:
                written += 1

    logging.info(
        f"Processing complete: {written} written, {skipped} skipped (outside mask), "
        f"{errors} errors out of {len(image_files)} total files."
    )


if __name__ == "__main__":
    parsed_args = parse_arguments()
    main(parsed_args)
