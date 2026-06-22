"""Run inference with a trained segmentation model to produce a vector line GeoPackage of predicted boundaries."""

import argparse
import logging
import os
import shutil
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime
import threading
import queue
import multiprocessing

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

import torch
import cv2
import numpy as np
import rasterio
from scipy.ndimage import convolve
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
    Suppress C-level stderr output (e.g. libtiff warnings) by temporarily redirecting
    file descriptor 2 to /dev/null. Python-level stderr is unaffected.
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

    logging.info(f"Architecture: {arch_name}")
    logging.info(f"Encoder: {encoder_name}")

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


def _writer_worker(write_queue, temp_dir):
    """
    Background thread worker that consumes (prob_map, chip_path, ...) items from
    write_queue and writes them as single-band float32 GeoTIFF files, decoupling
    disk I/O from GPU inference.
    """
    while True:
        item = write_queue.get()
        if item is None:  # sentinel: inference is done
            write_queue.put(None)  # Pass poison pill to next worker
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


def predict_chips(
    model, input_dir, temp_dir, device, batch_size=32, num_workers=4, use_tta=False
):
    """
    Run batched inference on all chips in input_dir and save to temp_dir.

    Args:
        batch_size: Number of chips per GPU forward pass.
        num_workers: CPU workers for DataLoader prefetch.
    """
    transform = get_preprocessing()
    chip_files = sorted(input_dir.glob("*.tif"))

    if not chip_files:
        logging.warning(f"No .tif files found in {input_dir}")
        return []

    logging.info(
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
    num_writers = 3
    writer_threads = []
    for _ in range(num_writers):
        t = threading.Thread(
            target=_writer_worker, args=(write_queue, temp_dir), daemon=True
        )
        t.start()
        writer_threads.append(t)

    output_files = []
    use_amp = device == "cuda"

    with torch.no_grad():
        for img_tensors, chip_paths, trans_tuples, crs_strs, valids in tqdm(
            loader, desc="Inference"
        ):
            img_tensors = img_tensors.to(device, non_blocking=True)

            with torch.amp.autocast(device, enabled=use_amp):
                if use_tta:
                    # Original prediction
                    logits1 = model(img_tensors)
                    probs1 = torch.sigmoid(logits1)

                    # 180 degree rotated prediction (k=2)
                    imgs_rot = torch.rot90(img_tensors, 2, [2, 3])
                    logits2 = model(imgs_rot)
                    probs2 = torch.sigmoid(logits2)

                    # Un-rotate
                    probs2_unrot = torch.rot90(probs2, -2, [2, 3])

                    # Take the element-wise maximum
                    probs_max = torch.maximum(probs1, probs2_unrot)
                    prob_maps = probs_max.squeeze(1).cpu().numpy()
                else:
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
    for t in writer_threads:
        t.join()

    return output_files


def _build_vrt_worker(args):
    vrt_path, input_files = args
    try:
        options = gdal.BuildVRTOptions(
            resampleAlg=gdal.GRA_NearestNeighbour, resolution="highest"
        )
        input_strs = [str(f) for f in input_files]
        gdal.BuildVRT(str(vrt_path), input_strs, options=options)
        return str(vrt_path)
    except Exception as e:
        logging.error(f"Worker VRT error: {e}")
        return None


def build_vrt(vrt_path, input_files):
    """Build a VRT from a list of input files."""
    logging.info(f"Building VRT from {len(input_files)} files...")

    if len(input_files) < 2000:
        try:
            options = gdal.BuildVRTOptions(
                resampleAlg=gdal.GRA_NearestNeighbour, resolution="highest"
            )
            input_strs = [str(f) for f in input_files]
            gdal.BuildVRT(str(vrt_path), input_strs, options=options)
            return True
        except Exception as e:
            logging.error(f"Error building VRT: {e}")
            return False

    # For massive datasets, parallelise the I/O bottleneck of reading thousands of TIFF headers
    chunk_size = 2000
    chunks = [
        input_files[i : i + chunk_size] for i in range(0, len(input_files), chunk_size)
    ]

    # Place intermediate VRTs alongside the input predictions for automatic cleanup
    temp_dir = Path(input_files[0]).parent / "temp_vrts"
    temp_dir.mkdir(exist_ok=True)

    worker_args = []
    for i, chunk in enumerate(chunks):
        temp_vrt = temp_dir / f"chunk_{i}.vrt"
        worker_args.append((temp_vrt, chunk))

    logging.info(
        f"Splitting VRT generation into {len(chunks)} chunks across multiple CPU cores..."
    )
    try:
        intermediate_vrts = []
        num_workers = max(1, multiprocessing.cpu_count() - 2)
        with multiprocessing.Pool(num_workers) as pool:
            for res in pool.imap_unordered(_build_vrt_worker, worker_args):
                if res:
                    intermediate_vrts.append(res)

        logging.info("Merging intermediate VRTs into final mosaic...")
        options = gdal.BuildVRTOptions(
            resampleAlg=gdal.GRA_NearestNeighbour, resolution="highest"
        )
        gdal.BuildVRT(str(vrt_path), intermediate_vrts, options=options)

        return True
    except Exception as e:
        logging.error(f"Error building VRT: {e}")
        return False


def process_chunk_worker(args):
    """
    Multiprocessing worker that reads one spatial chunk of the prediction VRT,
    applies morphological closing, skeletonises the binary mask, removes junction
    pixels, and returns a list of vectorised line coordinate arrays.
    """
    (
        vrt_path,
        col_start,
        row_start,
        width,
        height,
        chunk_size,
        threshold,
        min_contour_length,
        transform_tuple,
    ) = args
    transform = Affine(*transform_tuple)

    with suppress_stderr():
        with rasterio.open(vrt_path) as src:
            chunk_height = min(chunk_size, height - row_start)
            chunk_width = min(chunk_size, width - col_start)
            window = Window(col_start, row_start, chunk_width, chunk_height)
            chunk = src.read(1, window=window)

            # Skip empty chunks to save processing time and memory
            if not np.any(chunk > threshold):
                return []

            binary_chunk = (chunk > threshold).astype(np.uint8)

            # 1. Smooth the mask to prevent hairy spurs
            kernel_close = np.ones((3, 3), np.uint8)
            binary_chunk = cv2.morphologyEx(binary_chunk, cv2.MORPH_CLOSE, kernel_close)

            # 2. Skeletonise
            skeleton_chunk = skeletonize(binary_chunk).astype(np.uint8)

            # 3. Junction breaker: pixels with >2 connected neighbours are branch points.
            # Convolve with a flat 3x3 kernel — each skeleton pixel accumulates itself + neighbours.
            # Sum > 3 means the pixel (value 1) has 3+ skeleton neighbours, i.e. it is a junction.
            neighbors = convolve(
                skeleton_chunk, np.ones((3, 3), dtype=np.uint8), mode="constant", cval=0
            )
            junctions = (neighbors > 3) & (skeleton_chunk == 1)
            skeleton_chunk[junctions] = 0

            contours, _ = cv2.findContours(
                skeleton_chunk, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )

            chunk_lines = []
            for cnt in contours:
                if len(cnt) < min_contour_length:
                    continue
                coords_pix = cnt.squeeze().astype(float)
                if len(coords_pix.shape) != 2:
                    continue

                # Vectorized affine transformation relative to VRT start
                xs = coords_pix[:, 0] + col_start
                ys = coords_pix[:, 1] + row_start
                xs_map, ys_map = transform * (xs + 0.5, ys + 0.5)
                coords_map = np.column_stack((xs_map, ys_map))

                if len(coords_map) >= 2:
                    chunk_lines.append(coords_map)

            return chunk_lines


def process_vrt_to_lines(
    vrt_path, chunk_size=2048, threshold=0.5, min_contour_length=5
):
    """Process VRT in chunks to create skeleton and vectorize."""
    logging.info("Opening VRT for chunked processing...")

    with rasterio.open(vrt_path) as src:
        width = src.width
        height = src.height
        transform = src.transform
        crs = src.crs

        logging.info(f"Mosaic dimensions: {width} x {height} pixels")

        # calculate number of chunks
        n_chunks_x = int(np.ceil(width / chunk_size))
        n_chunks_y = int(np.ceil(height / chunk_size))
        total_chunks = n_chunks_x * n_chunks_y

        # Prepare arguments for parallel processing
        chunk_args = []
        transform_tuple = (
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f,
        )
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
                        min_contour_length,
                        transform_tuple,
                    )
                )

        lines = []
        num_workers = max(1, multiprocessing.cpu_count() - 2)
        logging.info(
            f"Processing in {chunk_size}x{chunk_size} chunks with {num_workers} workers..."
        )

        with multiprocessing.Pool(num_workers) as pool:
            for result_lines in tqdm(
                pool.imap_unordered(process_chunk_worker, chunk_args),
                total=len(chunk_args),
                desc="Skeletonizing & Vectorizing",
            ):
                for coords_map in result_lines:
                    lines.append(LineString(coords_map))

        return lines, crs


def extend_line(line, distance=0.5):
    """
    Extends a Shapely LineString at both ends by a specified distance.
    It calculates the trajectory of the first and last segments to ensure
    the extension is a straight continuation of the line.
    """
    if line is None or line.is_empty or len(line.coords) < 2:
        return line

    coords = list(line.coords)

    p1_start = np.array(coords[0])
    p2_start = np.array(coords[1])
    vector_start = p1_start - p2_start
    length_start = np.linalg.norm(vector_start)
    new_start = (
        p1_start + (vector_start / length_start) * distance
        if length_start > 0
        else p1_start
    )

    p1_end = np.array(coords[-2])
    p2_end = np.array(coords[-1])
    vector_end = p2_end - p1_end
    length_end = np.linalg.norm(vector_end)
    new_end = (
        p2_end + (vector_end / length_end) * distance if length_end > 0 else p2_end
    )

    new_coords = [tuple(new_start)] + coords + [tuple(new_end)]
    return LineString(new_coords)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Inference and Stitch Results")

    parser.add_argument(
        "--chip-dir",
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
        "--tta",
        action="store_true",
        help="Enable Test Time Augmentation (180 degree rotation + max pooling).",
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
        help="Probability threshold for converting the model output to a binary mask. Default: 0.5.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Tile size in pixels used when converting the prediction probability mosaic to vector lines. Larger values use more memory but may be faster. Default: 2048.",
    )
    parser.add_argument(
        "--min-contour-length",
        type=int,
        default=5,
        help="Minimum number of vertices a predicted line segment must have to be kept. Shorter segments are discarded. Default: 5.",
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
    parser.add_argument(
        "--extend-lines",
        type=float,
        default=0.5,
        help="Extend each predicted line at both ends by this distance in metres. Small extensions help connect lines that stop just short of a junction. Set to 0 to disable. Default: 0.5.",
    )
    parser.add_argument(
        "--prediction-mask",
        type=Path,
        default=None,
        help="Path to the prediction mask file (.gpkg or .shp). Output prediction lines are "
        "clipped to the mask before saving.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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
                logging.info(f"No model specified. Using most recent: {args.model}")

    if not args.model or not args.model.exists():
        logging.error(f"Error: Model not found: {args.model}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = args.output_dir / "temp_preds"

    use_existing = False
    if temp_dir.exists() and any(temp_dir.iterdir()):
        while True:
            resp = (
                input(
                    f"Directory '{temp_dir}' already exists and contains files. Use existing predictions (u) or overwrite (o)? [u/o]: "
                )
                .strip()
                .lower()
            )
            if resp in ["u", "o"]:
                break
            logging.warning("Please enter 'u' to use existing or 'o' to overwrite.")

        if resp == "u":
            use_existing = True
            logging.info("Skipping inference, using existing predictions...")
        else:
            logging.info("Overwriting existing predictions...")
            shutil.rmtree(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir.mkdir(parents=True, exist_ok=True)

    if use_existing:
        pred_files = list(temp_dir.glob("*.tif"))
    else:
        # 2. Load Model
        logging.info(f"Loading model from {args.model}...")
        model = load_model(args.model, device)

        if device == "cuda":
            try:
                model = torch.compile(model)
                logging.info("Model compiled with torch.compile()")
            except Exception as e:
                logging.warning(f"torch.compile() unavailable, skipping: {e}")

        # 3. Predict
        logging.info("Starting inference...")
        pred_files = predict_chips(
            model,
            args.chip_dir,
            temp_dir,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_tta=args.tta,
        )

    if not pred_files:
        logging.warning("No predictions generated. Exiting.")
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

    # 6. Extend lines
    if args.extend_lines > 0:
        lines = [extend_line(line, args.extend_lines) for line in lines]

    # 7. Clip to prediction mask (if provided)
    if args.prediction_mask is not None:
        if crs is None:
            raise ValueError(
                "Prediction output CRS is undefined. Cannot reproject prediction mask."
            )
        mask_path = args.prediction_mask.resolve()
        if not mask_path.exists():
            raise ValueError(f"Prediction mask not found: {mask_path}")
        mask_gdf = gpd.read_file(mask_path)
        if mask_gdf.empty:
            raise ValueError(f"Prediction mask contains no features: {mask_path}")
        mask_gdf = mask_gdf.to_crs(crs)
        mask_union = mask_gdf.union_all()
        if lines:
            gdf_lines = gpd.GeoDataFrame(geometry=lines, crs=crs)
            gdf_lines = gpd.clip(gdf_lines, mask_union)
            lines = list(gdf_lines.geometry)
            logging.info(
                f"Clipped prediction lines to mask: {len(lines)} lines remaining."
            )
            if not lines:
                logging.warning("All lines were outside the prediction mask.")

    # 8. Save Output
    if lines:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = args.model.stem
        out_gpkg = args.output_dir / f"{timestamp}_{model_name}_boundaries.gpkg"

        logging.info(f"Saving {len(lines)} boundaries to {out_gpkg}...")
        gdf = gpd.GeoDataFrame(geometry=lines, crs=crs)
        gdf.to_file(out_gpkg, driver="GPKG")
        logging.info("Done.")
    else:
        logging.warning("No lines detected.")

    # 9. Cleanup
    if not args.keep_preds:
        logging.info("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
        if vrt_path.exists():
            vrt_path.unlink()
    else:
        logging.info(f"Temporary files retained in {temp_dir}")


if __name__ == "__main__":
    main()
