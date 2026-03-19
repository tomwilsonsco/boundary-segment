import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling, ColorInterp
from tqdm import tqdm

from create_vrt import main as create_vrt_main


def parse_arguments(args=None):
    """Set up and parse command line arguments."""
    parser = argparse.ArgumentParser(description="Add upscaled NIR band to RGB images.")
    parser.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="Path to the directory containing target RGB images",
    )
    parser.add_argument(
        "--source-nir-dir",
        type=Path,
        required=True,
        help="Path to the directory containing source NIR images",
    )
    parser.add_argument(
        "--nir-band",
        type=int,
        default=1,
        help="Band number to read from the NIR VRT (default: 4)",
    )
    return parser.parse_args(args)


def main(args):
    target_dir = args.target_dir.resolve()
    source_nir_dir = args.source_nir_dir.resolve()

    if not target_dir.exists():
        raise ValueError(f"Target directory not found: {target_dir}")
    if not source_nir_dir.exists():
        raise ValueError(f"Source NIR directory not found: {source_nir_dir}")

    target_images = list(target_dir.glob("*.tif"))
    if not target_images:
        raise ValueError(f"No TIFF images found in {target_dir}")

    # a) Read first raster to count bands and pixel size
    first_image = target_images[0]
    with rasterio.open(first_image) as src:
        band_count = src.count
        pixel_size_x = src.transform.a
        pixel_size_y = abs(src.transform.e)

    if band_count not in [3, 4]:
        raise ValueError(
            f"Expected 3 or 4 bands in target images, but found {band_count}."
        )

    print(
        f"First target image has {band_count} bands. Pixel size: {pixel_size_x} x {pixel_size_y}"
    )

    # b) Use create_vrt script to make a VRT of all images in the source NIR directory
    vrt_filename = "nir_source.vrt"
    vrt_path = source_nir_dir / vrt_filename
    nir_crs = "EPSG:27700"

    print("Creating VRT for source NIR images...")
    vrt_args = argparse.Namespace(
        img_dir=source_nir_dir,
        output_filename=vrt_filename,
        crs=nir_crs,
    )
    create_vrt_main(vrt_args)

    if not vrt_path.exists():
        print("Error: VRT creation failed.", file=sys.stderr)
        sys.exit(1)

    # c) Iterate through each image in the target directory
    print("Processing target images and adding NIR band...")
    with rasterio.open(vrt_path) as nir_vrt:
        for target_img_path in tqdm(target_images, desc="Adding NIR band"):
            with rasterio.open(target_img_path) as tgt_src:
                # Skip if already processed to 4 bands
                if tgt_src.count == 4:
                    continue
                if tgt_src.count != 3:
                    print(
                        f"\nSkipping {target_img_path.name}: Expected 3 bands, got {tgt_src.count}"
                    )
                    continue

                tgt_profile = tgt_src.profile.copy()
                tgt_bounds = tgt_src.bounds
                tgt_width = tgt_src.width
                tgt_height = tgt_src.height
                tgt_data = tgt_src.read()

            # Calculate the image window in the NIR VRT
            nir_window = from_bounds(
                tgt_bounds.left,
                tgt_bounds.bottom,
                tgt_bounds.right,
                tgt_bounds.top,
                transform=nir_vrt.transform,
            )

            # Read windowed array and upscale using bilinear resampling
            nir_data = nir_vrt.read(
                args.nir_band,
                window=nir_window,
                out_shape=(tgt_height, tgt_width),
                resampling=Resampling.bilinear,
                boundless=True,
                fill_value=0,
            )

            # Update the metadata for 4 bands
            tgt_profile.update(count=4)
            tgt_profile.update(compress="lzw")
            tgt_profile["photometric"] = "rgb"
            tgt_profile["alpha"] = "unspecified"
            tgt_profile.pop("jpeg_quality", None)

            # Combine RGB and NIR
            new_data = np.zeros((4, tgt_height, tgt_width), dtype=tgt_profile["dtype"])
            new_data[0:3, :, :] = tgt_data
            new_data[3, :, :] = nir_data

            # Write to a temporary file and replace to avoid corruption
            temp_img_path = target_img_path.with_name(f"{target_img_path.stem}_tmp.tif")
            with rasterio.open(temp_img_path, "w", **tgt_profile) as dst:
                dst.write(new_data)
                dst.colorinterp = [
                    ColorInterp.red,
                    ColorInterp.green,
                    ColorInterp.blue,
                    ColorInterp.undefined,
                ]

            temp_img_path.replace(target_img_path)

    print("NIR bands added successfully!")


if __name__ == "__main__":
    parsed_args = parse_arguments()
    main(parsed_args)
