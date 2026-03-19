import argparse
from pathlib import Path
import tempfile
import multiprocessing
from functools import partial
import geopandas as gpd
from tqdm import tqdm
import rasterio
from rasterio import features
import numpy as np
from scipy.ndimage import distance_transform_edt


def process_mask_creation(chip_path, out_dir, features_path, buffer_size):
    """Worker function to create a single mask."""
    try:
        with rasterio.open(chip_path) as src:
            transform = src.transform
            crs = src.crs
            width = src.width
            height = src.height
            chip_bounds = src.bounds
            pixel_size = abs(transform.a)

        # Read only intersecting features to save computation
        lines_gdf = gpd.read_file(features_path, bbox=tuple(chip_bounds))
        
        # Maximum distance in pixels for the gradient decay
        max_dist_pixels = buffer_size / pixel_size

        if lines_gdf.empty:
            gradient_mask = np.zeros((height, width), dtype=np.float32)
        else:
            geometries = lines_gdf.geometry.tolist()
            rasterized = features.rasterize(
                geometries,
                out_shape=(height, width),
                transform=transform,
                fill=1,
                default_value=0,
                dtype=np.uint8
            )
            
            if np.all(rasterized == 1):
                # Fallback if lines intersected bbox but didn't touch the pixel grid
                gradient_mask = np.zeros((height, width), dtype=np.float32)
            else:
                distance = distance_transform_edt(rasterized)
                clipped_distance = np.clip(distance, 0, max_dist_pixels)
                gradient_mask = 1.0 - (clipped_distance / max_dist_pixels)
                gradient_mask = gradient_mask.astype(np.float32)

        out_path = out_dir / chip_path.name
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': 'float32',
            'crs': crs,
            'transform': transform,
            'compress': 'lzw'
        }
        
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(gradient_mask, 1)
            
        return True
    except Exception as e:
        print(f"Error processing {chip_path.name}: {e}")
        return False


def parse_arguments(args=None):
    """Set up and parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare segmentation masks.")
    parser.add_argument(
        "--chip-dir",
        type=Path,
        required=True,
        help="Path to the folder containing chip images",
    )
    parser.add_argument(
        "--shapefile",
        type=Path,
        required=True,
        help="Path to the land parcels shapefile or gpkg",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="masks",
        help="Name of the output subfolder (default: masks)",
    )
    parser.add_argument(
        "--buffer-size",
        type=float,
        default=0.75,
        help="Buffer size for lines in meters (default: 0.75)",
    )
    parser.add_argument(
        "--singleprocessor",
        action="store_true",
        help="Use single process instead of multiprocessing",
    )
    return parser.parse_args(args)


def main(args):
    """Main orchestration function."""
    chip_dir = args.chip_dir.resolve()
    shapefile = args.shapefile.resolve()

    if not chip_dir.exists():
        raise ValueError(f"Chip directory not found: {chip_dir}")
    if not shapefile.exists():
        raise ValueError(f"Shapefile not found: {shapefile}")

    out_dir = chip_dir / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading vector data...")
    gdf = gpd.read_file(shapefile)

    print("Converting polygons to lines...")
    lines = gdf.geometry.boundary

    print("Dissolving geometry (this may take a moment)...")
    lines = lines.explode(index_parts=True).union_all()

    # Store the 1D lines without buffering
    line_gdf = gpd.GeoDataFrame(geometry=[lines], crs=gdf.crs)

    chip_paths = list(chip_dir.glob("*.tif"))
    print(f"Processing {len(chip_paths)} chips...")

    tmp_file = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
    temp_gpkg_path = Path(tmp_file.name)
    tmp_file.close()  # release lock on tempfile (windows)

    success_count = 0
    try:
        line_gdf.to_file(temp_gpkg_path, driver="GPKG")

        if not args.singleprocessor:
            # use available cores - 1
            num_workers = max(1, multiprocessing.cpu_count() - 1)
            print(f"Using {num_workers} workers for processing.")

            func = partial(
                process_mask_creation, out_dir=out_dir, features_path=temp_gpkg_path, buffer_size=args.buffer_size
            )

            with multiprocessing.Pool(num_workers) as pool:
                results = list(
                    tqdm(
                        pool.imap_unordered(func, chip_paths),
                        total=len(chip_paths),
                        desc="Generating masks",
                    )
                )
            success_count = sum(results)
        else:
            print("Using single process.")
            for chip_path in tqdm(chip_paths, desc="Generating masks"):
                if process_mask_creation(chip_path, out_dir, temp_gpkg_path, args.buffer_size):
                    success_count += 1
    finally:
        # clean up the temporary file
        if temp_gpkg_path.exists():
            temp_gpkg_path.unlink()

    failed_count = len(chip_paths) - success_count
    print(
        f"Mask generation complete. Succeeded: {success_count}, Failed: {failed_count}"
    )


if __name__ == "__main__":
    parsed_args = parse_arguments()
    main(parsed_args)
