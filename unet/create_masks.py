import argparse
from pathlib import Path
import tempfile
import multiprocessing
from functools import partial
import geopandas as gpd
from tqdm import tqdm
from rschip import SegmentationMask


def process_mask_creation(chip_path, out_dir, features_path):
    """Worker function to create a single mask."""
    try:
        out_path = out_dir / chip_path.name
        mask = SegmentationMask(
            input_image_path=chip_path,
            input_features_path=features_path,
            output_path=out_path,
        )
        mask.create_mask(silent=True)
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
        "--parcels",
        type=Path,
        required=True,
        help="Path to the land parcels file (.gpkg or .shp accepted)",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="masks",
        help="Name of the output subfolder (default: masks)",
    )
    parser.add_argument(
        "--buffer-dist",
        type=float,
        default=0.75,
        help="Buffer distance for lines in meters (default: 0.75)",
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
    parcels = args.parcels.resolve()

    if not chip_dir.exists():
        raise ValueError(f"Chip directory not found: {chip_dir}")
    if not parcels.exists():
        raise ValueError(f"Parcels file not found: {parcels}")

    out_dir = chip_dir / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading vector data...")
    gdf = gpd.read_file(parcels)

    print("Converting polygons to lines...")
    lines = gdf.geometry.boundary

    print("Dissolving geometry (this may take a moment)...")
    lines = lines.explode(index_parts=True).union_all()

    print(f"Buffering lines by {args.buffer_dist}m...")
    buffered_geom = lines.buffer(args.buffer_dist)

    buffer_gdf = gpd.GeoDataFrame(geometry=[buffered_geom], crs=gdf.crs)

    # needed for rschip.SegmentationMask
    buffer_gdf["ml_class"] = 1

    chip_paths = list(chip_dir.glob("*.tif"))
    print(f"Processing {len(chip_paths)} chips...")

    tmp_file = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
    temp_gpkg_path = Path(tmp_file.name)
    tmp_file.close()  # release lock on tempfile (windows)

    success_count = 0
    try:
        buffer_gdf.to_file(temp_gpkg_path, driver="GPKG")

        if not args.singleprocessor:
            # use available cores - 1
            num_workers = max(1, multiprocessing.cpu_count() - 1)
            print(f"Using {num_workers} workers for processing.")

            func = partial(
                process_mask_creation, out_dir=out_dir, features_path=temp_gpkg_path
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
                if process_mask_creation(chip_path, out_dir, temp_gpkg_path):
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
