import rasterio
from rasterio.enums import Resampling
import multiprocessing
from pathlib import Path
from tqdm import tqdm
import argparse
from functools import partial


def process_func(file_path, out_dir, scale_factor):
    output_file = out_dir / f"{file_path.stem}_ds.tif"
    with rasterio.open(file_path) as src:
        # calculate new dimensions
        new_height = int(src.height * scale_factor)
        new_width = int(src.width * scale_factor)

        # resample data to target shape
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.bilinear,
        )

        transform = src.transform * src.transform.scale(
            (src.width / data.shape[-1]), (src.height / data.shape[-2])
        )

        profile = src.profile.copy()
        profile.update(
            {"transform": transform, "width": new_width, "height": new_height}
        )
        profile.update({"compress": "LZW", "photometric": "RGB", "tiled": True})
        profile.pop("jpeg_quality", None)

        with rasterio.open(output_file, "w", **profile) as dst:
            dst.write(data)


def parse_arguments(args=None):
    """Set up and parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Downscale geotiff images by a specific factor."
    )
    parser.add_argument(
        "--img-dir",
        type=Path,
        required=True,
        help="Path to the folder containing geotiffs",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        help="Name of the output subfolder to write downscaled images",
    )
    parser.add_argument(
        "--downscale-factor",
        default=2,
        type=int,
        help="1/<input number> will be how much image is downscaled by."
        "For example setting 2 will convert 0.125m per pixel images to 0.25m",
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

    if not img_dir.exists():
        raise ValueError(f"Folder not found: {img_dir}")

    image_files = list(img_dir.glob("*.tif"))

    if not image_files:
        raise ValueError(f"No tif images found to process in {img_dir}")

    subdir = args.output_subdir if args.output_subdir else "downscaled"

    out_dir = img_dir / subdir
    out_dir.mkdir(exist_ok=True)

    scale_factor = 1 / args.downscale_factor

    print(f"Found {len(image_files)} TIFF files in {img_dir}")

    if not args.singleprocessor:
        # Use available cores - 1
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        print(f"Using {num_workers} workers for processing.")

        func = partial(process_func, out_dir=out_dir, scale_factor=scale_factor)

        with multiprocessing.Pool(num_workers) as pool:
            list(
                tqdm(
                    pool.imap_unordered(func, image_files),
                    total=len(image_files),
                    desc="Downscaling geotiffs",
                )
            )
    else:
        print("Using single process.")
        for file_path in tqdm(image_files, desc="Downscaling geotiffs"):
            process_func(file_path, out_dir, scale_factor)


if __name__ == "__main__":
    parsed_args = parse_arguments()
    main(parsed_args)
