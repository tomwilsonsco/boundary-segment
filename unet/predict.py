import argparse
import os
import shutil
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime
import threading
import queue
import multiprocessing

import torch
import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
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

    # remove '_orig_mod.' prefix if model was trained with torch.compile
    clean_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k
        clean_state_dict[new_k] = v
    state_dict = clean_state_dict

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
    elif arch_name == "fpn":
        model = smp.FPN(
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


class ChipInferenceDataset(torch.utils.data.Dataset):
    """
    Minimal Dataset for inference — reads chips and returns tensor + profile metadata.
    Storing the profile per-chip is cheap (it's just a small dict).
    """

    def __init__(self, chip_files, transform):
        self.chip_files = chip_files
        self.transform = transform

    def __len__(self):
        return len(self.chip_files)

    def __getitem__(self, idx):
        chip_path = self.chip_files[idx]

        # Defaults
        image = None
        trans_tuple = (0.0,) * 6
        crs_str = ""

        with suppress_stderr():
            try:
                with rasterio.open(chip_path) as src:
                    image = src.read()
                    image = np.moveaxis(image, 0, -1)  # (C,H,W) -> (H,W,C)
                    t = src.transform
                    trans_tuple = (t.a, t.b, t.c, t.d, t.e, t.f)
                    crs_str = src.crs.to_string() if src.crs else ""
            except Exception:
                image = None

        if image is None:
            h, w = 128, 128
            return torch.zeros(3, h, w), str(chip_path), trans_tuple, crs_str, False

        augmented = self.transform(image=image)
        img_tensor = augmented["image"]  # (C, H, W)
        return img_tensor, str(chip_path), trans_tuple, crs_str, True


def _writer_worker(write_queue, input_dir, temp_dir):
    """
    Runs in a background thread — drains (prob_map, chip_path) pairs from
    the queue and writes GeoTIFF prediction files without blocking inference.
    """
    while True:
        item = write_queue.get()
        if item is None:  # sentinel: inference is done
            break
        prob_map, chip_path_str, trans_tuple, crs_str, valid = item
        if not valid:
            write_queue.task_done()
            continue

        chip_path = Path(chip_path_str)
        out_path = temp_dir / chip_path.name

        # Reconstruct affine transform
        transform = Affine(*trans_tuple)

        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "height": prob_map.shape[0],
            "width": prob_map.shape[1],
            "transform": transform,
            "crs": crs_str,
            "compress": "lzw",
            "nodata": 0,
        }

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(prob_map, 1)

        write_queue.task_done()


def predict_chips(model, input_dir, temp_dir, device, batch_size=32, num_workers=4):
    """
    Run batched inference on all chips in input_dir and save to temp_dir.

    Args:
        batch_size: Number of chips per GPU forward pass.
        num_workers: CPU workers for DataLoader prefetch.
    """
    transform = get_preprocessing()
    chip_files = sorted(input_dir.glob("*.tif"))

    if not chip_files:
        print(f"No .tif files found in {input_dir}")
        return []

    print(
        f"Predicting on {len(chip_files)} chips "
        f"(batch_size={batch_size}, num_workers={num_workers})..."
    )

    dataset = ChipInferenceDataset(chip_files, transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )

    # Background writer thread — decouples GPU inference from disk writes
    write_queue = queue.Queue(maxsize=batch_size * 4)
    writer_thread = threading.Thread(
        target=_writer_worker,
        args=(write_queue, input_dir, temp_dir),
        daemon=True,
    )
    writer_thread.start()

    output_files = []
    use_amp = device == "cuda"

    with torch.no_grad():
        for img_tensors, chip_paths, trans_tuples, crs_strs, valids in tqdm(
            loader, desc="Inference"
        ):
            img_tensors = img_tensors.to(device, non_blocking=True)

            with torch.amp.autocast(device, enabled=use_amp):
                logits = model(img_tensors)
                prob_maps = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                # squeeze(1): (B,1,H,W) -> (B,H,W)

            # Iterate through batch
            for i, (prob_map, chip_path_str, valid) in enumerate(
                zip(prob_maps, chip_paths, valids)
            ):
                valid_bool = valid.item() if isinstance(valid, torch.Tensor) else valid

                # trans_tuples is a list of 6 tensors (components), access ith element of each
                # crs_strs is a tuple of strings
                t_tup = tuple(trans_tuples[k][i].item() for k in range(6))
                c_str = crs_strs[i]

                write_queue.put((prob_map, chip_path_str, t_tup, c_str, valid_bool))

                if valid_bool:
                    output_files.append(temp_dir / Path(chip_path_str).name)

    # Signal writer to finish and wait for all writes to complete
    write_queue.put(None)
    writer_thread.join()

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


def process_chunk_worker(args):
    """Worker function for parallel VRT processing."""
    vrt_path, col_start, row_start, width, height, chunk_size, threshold = args
    with suppress_stderr():
        with rasterio.open(vrt_path) as src:
            chunk_height = min(chunk_size, height - row_start)
            chunk_width = min(chunk_size, width - col_start)
            window = Window(col_start, row_start, chunk_width, chunk_height)
            chunk = src.read(1, window=window)
            binary_chunk = (chunk > threshold).astype(np.uint8)
            skeleton_chunk = skeletonize(binary_chunk).astype(np.uint8)
            return (row_start, col_start, chunk_height, chunk_width, skeleton_chunk)


def process_vrt_to_lines(
    vrt_path, chunk_size=2048, threshold=0.5, min_contour_length=5
):
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

        # Prepare arguments for parallel processing
        chunk_args = []
        for row_start in range(0, height, chunk_size):
            for col_start in range(0, width, chunk_size):
                chunk_args.append(
                    (
                        str(vrt_path),
                        col_start,
                        row_start,
                        width,
                        height,
                        chunk_size,
                        threshold,
                    )
                )

        num_workers = max(1, multiprocessing.cpu_count() - 2)
        print(
            f"Processing in {chunk_size}x{chunk_size} chunks with {num_workers} workers..."
        )

        with multiprocessing.Pool(num_workers) as pool:
            for result in tqdm(
                pool.imap_unordered(process_chunk_worker, chunk_args),
                total=len(chunk_args),
                desc="Skeletonizing",
            ):
                r, c, h, w, skel = result
                full_skeleton[r : r + h, c : c + w] = skel

        print("Converting skeleton to vector lines...")
        lines = skeleton_to_lines(
            full_skeleton, transform, min_contour_length=min_contour_length
        )

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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference. Larger = faster GPU utilisation. Default: 32.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes for prefetching chips. Default: 4.",
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
    pred_files = predict_chips(
        model,
        args.input_dir,
        temp_dir,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

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
        min_contour_length=args.min_contour_length,
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
