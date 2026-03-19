import os
import shutil
import sys
import random
import subprocess
import argparse
import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import box
from tqdm import tqdm
from pathlib import Path
from contextlib import contextmanager


@contextmanager
def suppress_stderr():
    """
    Suppress C-level stderr output (like libtiff warnings).
    """
    try:
        null_fd = os.open(os.devnull, os.O_RDWR)
        save_fd = os.dup(2)
        os.dup2(null_fd, 2)
        yield
    except Exception:
        yield
    finally:
        try:
            os.dup2(save_fd, 2)
            os.close(null_fd)
            os.close(save_fd)
        except Exception:
            pass


def setup_directories(test_chips_dir, screenshot_output_dir):
    """Create necessary directories."""
    test_chips_dir.mkdir(parents=True, exist_ok=True)
    screenshot_output_dir.mkdir(parents=True, exist_ok=True)


def sample_and_copy_chips(src_dir, dst_dir, num_samples=50, seed=42):
    """Sample tif files and copy to test_chips directory.

    Returns:
        List of sampled filenames (basename only)
    """
    random.seed(seed)

    all_images = list(src_dir.glob("*.tif"))

    if len(all_images) < num_samples:
        print(f"Warning: Only {len(all_images)} images available, using all.")
        sampled = all_images
    else:
        sampled = random.sample(all_images, num_samples)

    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    sampled_basenames = []
    print(f"Copying {len(sampled)} chips to {dst_dir}...")
    for src_path in tqdm(sampled):
        basename = src_path.name
        dst_path = dst_dir / basename
        shutil.copy2(src_path, dst_path)
        sampled_basenames.append(basename)

    return sampled_basenames


def get_mask_paths(basenames, mask_dir):
    """Get corresponding mask paths for sampled images.

    Returns:
        Dict mapping basename to mask path
    """
    mask_paths = {}
    for basename in basenames:
        mask_path = mask_dir / basename
        if mask_path.exists():
            mask_paths[basename] = mask_path
        else:
            print(f"Warning: Mask not found for {basename}")
    return mask_paths


def load_parcels(parcels_path):
    """Load the land parcels.gpkg for AOI.

    Returns:
        GeoDataFrame of parcels
    """
    print(f"Loading parcels from {parcels_path}...")
    return gpd.read_file(parcels_path)


def run_unet_predict(input_dir, output_dir, model_path=None, scaler_path=None):
    """Run unet/predict_nir.py script."""
    print("\nRunning UNet prediction (NIR)...")

    cmd = [
        sys.executable,
        "unet/predict_nir.py",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--keep-preds",
    ]

    if model_path:
        cmd.extend(["--model", str(model_path)])
        
    if scaler_path:
        cmd.extend(["--scaler-path", str(scaler_path)])

    result = subprocess.run(cmd, text=True, capture_output=True)

    temp_pred_dir = output_dir / "temp_preds"
    if result.returncode != 0:
        pred_files = list(temp_pred_dir.glob("*.tif"))
        if len(pred_files) > 0:
            print(
                f"Warning: predict_nir.py exited with code {result.returncode}, but {len(pred_files)} predictions found. Continuing..."
            )
        else:
            print(
                f"Prediction process failed.\nStdout: {result.stdout}\nStderr: {result.stderr}"
            )
            raise RuntimeError(
                f"Prediction failed with return code {result.returncode} and no predictions found"
            )
    print("Prediction complete.")


def get_latest_output_gpkg(output_dir):
    """Get the most recently created output gpkg.

    Returns:
        Path to the output gpkg
    """
    gpkg_files = list(output_dir.glob("*_boundaries.gpkg"))
    if not gpkg_files:
        raise FileNotFoundError("No output gpkg files found")

    latest = max(gpkg_files, key=os.path.getmtime)
    return latest


def get_chip_bounds(chip_path):
    """Get the bounding box of a chip in CRS coordinates.

    Returns:
        Shapely box geometry
    """
    with suppress_stderr():
        with rasterio.open(chip_path) as src:
            bounds = src.bounds
            return box(bounds.left, bounds.bottom, bounds.right, bounds.top), src.crs


def filter_geometries_to_bounds(gdf, bounds_geom, crs):
    """Filter geodataframe to geometries that intersect bounds (no clipping).

    This avoids creating artificial boundary edges from clipping.

    Returns:
        GeoDataFrame filtered to intersecting geometries
    """
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    intersects_mask = gdf.geometry.intersects(bounds_geom)
    return gdf[intersects_mask].copy()


def create_6panel_plot(
    basename, chip_path, mask_path, pred_path, parcels_gdf, pred_lines_gdf, output_path
):
    """Create a 6-panel plot for a single chip.

    Panels:
    1. RGB chip
    2. RGB chip with parcels overlaid (thin red outline)
    3. Binary mask image
    4. Prediction probability map from TEMP_PRED_DIR
    5. Positive and negative difference between prediction and mask
    6. Prediction gpkg lines overlaid on RGB chip
    """
    with suppress_stderr():
        with rasterio.open(chip_path) as src:
            # Safely grab only the first 3 channels (RGB) for visualization
            rgb = src.read([1, 2, 3])
            rgb = np.moveaxis(rgb, 0, -1)
            transform = src.transform
            crs = src.crs
            bounds = src.bounds

    # normalise
    if rgb.dtype == np.uint8:
        rgb_display = rgb.astype(np.float32) / 255.0
    else:
        rgb_display = rgb.astype(np.float32)
        rgb_display = (rgb_display - rgb_display.min()) / (
            rgb_display.max() - rgb_display.min() + 1e-8
        )

    # read mask
    with suppress_stderr():
        with rasterio.open(mask_path) as src:
            mask = src.read(1)

    mask_binary = (mask > 0.5).astype(np.float32)

    if pred_path and os.path.exists(pred_path):
        with suppress_stderr():
            with rasterio.open(pred_path) as src:
                pred = src.read(1)
    else:
        pred = np.zeros_like(mask_binary)
        print(f"Warning: Prediction not found for {basename}")

    pred_binary = (pred > 0.5).astype(np.float32)

    bounds_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    parcels_filtered = filter_geometries_to_bounds(parcels_gdf.copy(), bounds_geom, crs)

    if pred_lines_gdf is not None and len(pred_lines_gdf) > 0:
        pred_lines_filtered = filter_geometries_to_bounds(
            pred_lines_gdf.copy(), bounds_geom, crs
        )
    else:
        pred_lines_filtered = gpd.GeoDataFrame(geometry=[], crs=crs)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # image dimensions for axis limits
    img_height, img_width = rgb_display.shape[:2]

    # 1. RGB chip
    ax1 = axes[0, 0]
    ax1.imshow(rgb_display)
    ax1.set_title("RGB Chip", fontsize=16)
    ax1.axis("off")

    # 2. RGB with parcels red outline
    ax2 = axes[0, 1]
    ax2.imshow(rgb_display)

    if len(parcels_filtered) > 0:
        for geom in parcels_filtered.geometry:
            if geom is not None and not geom.is_empty:
                if geom.geom_type == "Polygon":
                    x, y = geom.exterior.xy
                    pixels = [~transform * (xi, yi) for xi, yi in zip(x, y)]
                    px, py = zip(*pixels)
                    ax2.plot(px, py, "r-", linewidth=1.6, clip_on=True)
                elif geom.geom_type == "MultiPolygon":
                    for poly in geom.geoms:
                        x, y = poly.exterior.xy
                        pixels = [~transform * (xi, yi) for xi, yi in zip(x, y)]
                        px, py = zip(*pixels)
                        ax2.plot(px, py, "r-", linewidth=1.6, clip_on=True)

    ax2.set_xlim(0, img_width)
    ax2.set_ylim(img_height, 0)
    ax2.set_clip_on(True)
    ax2.set_title("RGB + True Parcel Boundaries", fontsize=16)
    ax2.axis("off")

    # 3. binary mask
    ax3 = axes[0, 2]
    ax3.imshow(mask_binary, cmap="gray", vmin=0, vmax=1)
    ax3.set_title("Ground Truth Mask", fontsize=16)
    ax3.axis("off")

    # 4. pred probability map
    ax4 = axes[1, 0]
    ax4.imshow(pred, cmap="gray", vmin=0, vmax=1)
    ax4.set_title("Prediction Probability", fontsize=16)
    ax4.axis("off")

    # 5. diff (positive/negative)
    ax5 = axes[1, 1]
    # Green = true positive (pred=1, mask=1)
    # Red = false positive (pred=1, mask=0)
    # Blue = false negative (pred=0, mask=1)
    diff_rgb = np.zeros((*mask_binary.shape, 3), dtype=np.float32)

    # tp
    true_pos = (mask_binary > 0.5) & (pred_binary > 0.5)
    diff_rgb[true_pos] = [0, 1, 0]  # green

    # fp
    false_pos = (mask_binary < 0.5) & (pred_binary > 0.5)
    diff_rgb[false_pos] = [1, 0, 0]  # red

    # fn
    false_neg = (mask_binary > 0.5) & (pred_binary < 0.5)
    diff_rgb[false_neg] = [0, 0, 1]  # blue

    ax5.imshow(diff_rgb)
    ax5.set_title("Difference: TP (green), FP (red), FN (blue)", fontsize=16)
    ax5.axis("off")

    # legend
    legend_elements = [
        Patch(facecolor="green", label="True Positive"),
        Patch(facecolor="red", label="False Positive"),
        Patch(facecolor="blue", label="False Negative"),
    ]
    ax5.legend(handles=legend_elements, loc="lower right", fontsize=8)

    # 6. RGB with prediction lines
    ax6 = axes[1, 2]
    ax6.imshow(rgb_display)

    if len(pred_lines_filtered) > 0:
        for geom in pred_lines_filtered.geometry:
            if geom is not None and not geom.is_empty:
                if geom.geom_type == "LineString":
                    x, y = geom.xy
                    pixels = [~transform * (xi, yi) for xi, yi in zip(x, y)]
                    px, py = zip(*pixels)
                    ax6.plot(px, py, "r-", linewidth=1.6, clip_on=True)
                elif geom.geom_type == "MultiLineString":
                    for line in geom.geoms:
                        x, y = line.xy
                        pixels = [~transform * (xi, yi) for xi, yi in zip(x, y)]
                        px, py = zip(*pixels)
                        ax6.plot(px, py, "r-", linewidth=1.6, clip_on=True)

    ax6.set_xlim(0, img_width)
    ax6.set_ylim(img_height, 0)
    ax6.set_clip_on(True)
    ax6.set_title("RGB + Predicted Boundary Lines", fontsize=16)
    ax6.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate UNet example screenshots (NIR)")

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("inputs/images/dataset"),
        help="Root dataset directory containing images/ and masks/ subdirs",
    )
    parser.add_argument(
        "--scaler-path",
        type=Path,
        default=None,
        help="Path to scaler.json containing mean and std values. Defaults to dataset_dir/scaler.json",
    )
    parser.add_argument(
        "--parcels-gpkg",
        type=Path,
        default=Path("inputs/aoi_parcels.gpkg"),
        help="Path to ground truth parcels GPKG",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to model checkpoint (optional, defaults to latest)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/unet_screenshots"),
        help="Directory to save screenshots",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("outputs/temp_inference"),
        help="Temporary directory for inference",
    )
    parser.add_argument(
        "--num-samples", type=int, default=50, help="Number of samples to plot"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip sampling and prediction, only regenerate plots from existing temp data",
    )

    return parser.parse_args()


def main(args):
    """Main function to generate example screenshots."""
    print("=" * 60)
    print("UNet Example Screenshot Generator (NIR)")
    if args.plots_only:
        print("(Plots only mode - using existing chips and predictions)")
    print("=" * 60)

    test_img_dir = args.dataset_dir / "images/test"
    test_mask_dir = args.dataset_dir / "masks/test"
    test_chips_dir = args.temp_dir / "test_chips"
    temp_pred_dir = args.temp_dir / "temp_preds"

    setup_directories(test_chips_dir, args.output_dir)

    if args.plots_only:
        print("\n[Step 1] Using existing chips in test_chips directory...")
        chip_files = list(test_chips_dir.glob("*.tif"))
        sampled_basenames = [f.name for f in chip_files]
        print(f"Found {len(sampled_basenames)} existing chips")
    else:
        print("\n[Step 1] Sampling and copying chips...")
        sampled_basenames = sample_and_copy_chips(
            test_img_dir, test_chips_dir, args.num_samples, args.seed
        )

    # mask paths
    print("\n[Step 2] Getting mask paths...")
    mask_paths = get_mask_paths(sampled_basenames, test_mask_dir)
    print(f"Found {len(mask_paths)} masks for {len(sampled_basenames)} chips")

    # parcels
    print("\n[Step 3] Loading parcels...")
    parcels_gdf = load_parcels(args.parcels_gpkg)
    print(f"Loaded {len(parcels_gdf)} parcels")

    if args.plots_only:
        print("\n[Step 4] Skipping prediction (plots-only mode)...")
    else:
        # pred
        scaler_file = args.scaler_path if args.scaler_path else args.dataset_dir / "scaler.json"
        if not scaler_file.exists():
            print(f"Warning: scaler config not found at {scaler_file}. predict_nir.py might fail.")
            scaler_file = None
            
        print("\n[Step 4] Running UNet prediction...")
        run_unet_predict(test_chips_dir, args.temp_dir, args.model, scaler_file)

    # files
    print("\n[Step 5] Getting prediction files...")
    pred_files = list(temp_pred_dir.glob("*.tif"))
    pred_map = {p.name: p for p in pred_files}
    print(f"Found {len(pred_map)} prediction files")

    # load output gpkg
    print("\n[Step 6] Loading prediction lines...")
    try:
        output_gpkg = get_latest_output_gpkg(args.temp_dir)
        print(f"Loading predictions from: {output_gpkg}")
        pred_lines_gdf = gpd.read_file(output_gpkg)
        print(f"Loaded {len(pred_lines_gdf)} predicted boundary lines")
    except FileNotFoundError:
        print(
            "Warning: No output gpkg found (merge may have failed). Plot 6 will show no pred lines."
        )
        pred_lines_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:27700")

    # plots
    print("\n[Step 7] Creating 6-panel plots...")
    successful = 0
    for basename in tqdm(sampled_basenames):
        if basename not in mask_paths:
            continue

        chip_path = test_chips_dir / basename
        mask_path = mask_paths[basename]
        pred_path = pred_map.get(basename)

        output_name = basename.replace(".tif", "_analysis.png")
        output_path = args.output_dir / output_name

        try:
            create_6panel_plot(
                basename=basename,
                chip_path=chip_path,
                mask_path=mask_path,
                pred_path=pred_path,
                parcels_gdf=parcels_gdf,
                pred_lines_gdf=pred_lines_gdf,
                output_path=output_path,
            )
            successful += 1
        except Exception as e:
            print(f"\nError processing {basename}: {e}")

    print("\n" + "=" * 60)
    print(f"Complete! Generated {successful} analysis plots.")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parsed_args = parse_arguments()
    main(parsed_args)
