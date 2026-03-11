import argparse
import os
import shutil
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime

import torch
import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from osgeo import gdal
import geopandas as gpd
from shapely.geometry import LineString
from skimage.morphology import skeletonize
from tqdm import tqdm
import segmentation_models_pytorch as smp
import albumentations as albu
from albumentations.pytorch import ToTensorV2


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


def get_preprocessing():
    """Get preprocessing transforms for inference."""
    return albu.Compose(
        [
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def load_model(model_path, device):
    """Load trained model from checkpoint with metadata support."""
    checkpoint = torch.load(model_path, map_location=device)

    state_dict = None
    arch_name = None
    encoder_name = None

    # Check for metadata in new format
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        arch_name = checkpoint.get("arch")
        encoder_name = checkpoint.get("encoder")
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        arch_name = checkpoint.get("arch")
        encoder_name = checkpoint.get("encoder")
    else:
        raise ValueError("Model checkpoint does not contain metadata (arch/encoder).")

    if arch_name is None or encoder_name is None:
        raise ValueError(
            f"Architecture or Encoder not found in checkpoint. Arch: {arch_name}, Encoder: {encoder_name}"
        )

    print(f"Architecture: {arch_name}")
    print(f"Encoder: {encoder_name}")

    if arch_name == "unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )
    elif arch_name == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )
    elif arch_name == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )
    else:
        raise ValueError(f"Unsupported architecture: {arch_name}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def predict_chips(model, input_dir, temp_dir, device):
    """Run inference on all chips in input_dir and save to temp_dir."""
    transform = get_preprocessing()
    chip_files = list(input_dir.glob("*.tif"))

    if not chip_files:
        print(f"No .tif files found in {input_dir}")
        return []

    print(f"Predicting on {len(chip_files)} chips...")
    output_files = []

    with torch.no_grad():
        for chip_path in tqdm(chip_files, desc="Inference"):
            with suppress_stderr():
                image = cv2.imread(str(chip_path))
            
            if image is None:
                print(f"Warning: Could not read {chip_path}")
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            with rasterio.open(chip_path) as src:
                profile = src.profile.copy()

            augmented = transform(image=image)
            img_tensor = augmented["image"].unsqueeze(0).to(device)


            logits = model(img_tensor)
            prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()

            out_path = temp_dir / chip_path.name
            profile.update(
                {
                    "dtype": "float32",
                    "count": 1,
                    "compress": "lzw",
                    "nodata": 0,
                }
            )

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(prob_map, 1)

            output_files.append(out_path)

    return output_files


def build_vrt(vrt_path, input_files):
    """Build a VRT from a list of input files."""
    print(f"Building VRT from {len(input_files)} files...")
    try:
        options = gdal.BuildVRTOptions(
            resampleAlg=gdal.GRA_NearestNeighbour, resolution="highest"
        )
        input_strs = [str(f) for f in input_files]
        gdal.BuildVRT(str(vrt_path), input_strs, options=options)
        return True
    except Exception as e:
        print(f"Error building VRT: {e}")
        return False


def skeleton_to_lines(skeleton, transform, min_contour_length=5):
    """Convert skeletonized binary mask to vector LineStrings."""
    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    lines = []

    print(f"Vectorizing {len(contours)} detected segments...")

    for cnt in tqdm(contours, desc="Vectorizing"):
        if len(cnt) < min_contour_length:
            continue
        coords_pix = cnt.squeeze().astype(float)
        if len(coords_pix.shape) != 2:
            continue

        coords_map = []
        for x_pix, y_pix in coords_pix:
            x_map, y_map = rasterio.transform.xy(
                transform, y_pix, x_pix, offset="center"
            )
            coords_map.append((x_map, y_map))

        if len(coords_map) >= 2:
            lines.append(LineString(coords_map))

    return lines


def process_vrt_to_lines(vrt_path, chunk_size=2048, threshold=0.5, min_contour_length=5):
    """Process VRT in chunks to create skeleton and vectorize."""
    print("Opening VRT for chunked processing...")

    with rasterio.open(vrt_path) as src:
        width = src.width
        height = src.height
        transform = src.transform
        crs = src.crs

        print(f"Mosaic dimensions: {width} x {height} pixels")
        
        # initialize full mask array (uint8 is efficient)
        full_skeleton = np.zeros((height, width), dtype=np.uint8)

        # calculate number of chunks
        n_chunks_x = int(np.ceil(width / chunk_size))
        n_chunks_y = int(np.ceil(height / chunk_size))
        total_chunks = n_chunks_x * n_chunks_y

        print(f"Processing in {chunk_size}x{chunk_size} chunks ({total_chunks} total)...")

        with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
            for row_start in range(0, height, chunk_size):
                for col_start in range(0, width, chunk_size):

                    chunk_height = min(chunk_size, height - row_start)
                    chunk_width = min(chunk_size, width - col_start)

                    window = Window(col_start, row_start, chunk_width, chunk_height)
                    chunk = src.read(1, window=window)

                    binary_chunk = (chunk > threshold).astype(np.uint8)
                    skeleton_chunk = skeletonize(binary_chunk).astype(np.uint8)

                    full_skeleton[
                        row_start : row_start + chunk_height,
                        col_start : col_start + chunk_width,
                    ] = skeleton_chunk

                    pbar.update(1)

        print("Converting skeleton to vector lines...")
        lines = skeleton_to_lines(full_skeleton, transform, min_contour_length=min_contour_length)

        return lines, crs


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Inference and Stitch Results")

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing input chip images (.tif).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to model checkpoint. Defaults to most recent in models/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/predictions"),
        help="Directory to save output GPKG. Default: outputs/predictions",
    )
    parser.add_argument(
        "--keep-preds",
        action="store_true",
        help="Retain intermediate prediction TIFF files. Default is to delete them.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for binary mask. Default: 0.5",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Chunk size for processing VRT. Default: 2048",
    )
    parser.add_argument(
        "--min-contour-length",
        type=int,
        default=5,
        help="Minimum number of vertices for a line prediction to be retained. Default: 5",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Setup Paths
    if args.model is None:
        models_dir = Path("models")
        if models_dir.exists():
            files = list(models_dir.glob("*.pth"))
            # Prefer inference files
            inference_files = [f for f in files if "_checkpoint.pth" not in f.name]
            candidates = inference_files if inference_files else files
            if candidates:
                candidates.sort(key=lambda f: f.name)
                args.model = candidates[-1]
                print(f"No model specified. Using most recent: {args.model}")
    
    if not args.model or not args.model.exists():
        print(f"Error: Model not found: {args.model}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = args.output_dir / "temp_preds"
    temp_dir.mkdir(exist_ok=True)

    # 2. Load Model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model, device)

    # 3. Predict
    print("Starting inference...")
    pred_files = predict_chips(model, args.input_dir, temp_dir, device)
    
    if not pred_files:
        print("No predictions generated. Exiting.")
        return

    # 4. Stitch (VRT)
    vrt_path = args.output_dir / "mosaic.vrt"
    if not build_vrt(vrt_path, pred_files):
        return

    # 5. Process VRT -> Skeleton -> Lines
    lines, crs = process_vrt_to_lines(
        vrt_path, 
        chunk_size=args.chunk_size, 
        threshold=args.threshold,
        min_contour_length=args.min_contour_length
    )

    # 6. Save Output
    if lines:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = args.model.stem
        out_gpkg = args.output_dir / f"{timestamp}_{model_name}_boundaries.gpkg"
        
        print(f"Saving {len(lines)} boundaries to {out_gpkg}...")
        gdf = gpd.GeoDataFrame(geometry=lines, crs=crs)
        gdf.to_file(out_gpkg, driver="GPKG")
        print("Done.")
    else:
        print("No lines detected.")

    # 7. Cleanup
    if not args.keep_preds:
        print("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
        if vrt_path.exists():
            vrt_path.unlink()
    else:
        print(f"Temporary files retained in {temp_dir}")


if __name__ == "__main__":
    main()