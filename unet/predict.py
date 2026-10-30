"""Run inference with a trained segmentation model to produce a vector line GeoPackage of predicted boundaries."""

import argparse
import logging
import multiprocessing
import os
import queue
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import albumentations as albu
import cv2
import geopandas as gpd
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import shapely
import torch
from albumentations.pytorch import ToTensorV2
from osgeo import gdal
from rasterio.transform import Affine
from rasterio.windows import Window
from shapely.geometry import LineString, MultiLineString
from skimage.morphology import skeletonize
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )
    elif arch_name == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )
    elif arch_name == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )
    elif arch_name == "fpn":
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=None,
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
        # size of the placeholder returned for unreadable chips, which must match
        # the other chips so the batch can still be stacked
        self.placeholder_shape = (128, 128)
        with suppress_stderr():
            for chip_path in chip_files:
                try:
                    with rasterio.open(chip_path) as src:
                        self.placeholder_shape = (src.height, src.width)
                    break
                except Exception:
                    continue

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
            h, w = self.placeholder_shape
            return torch.zeros(3, h, w), str(chip_path), trans_tuple, crs_str, False

        augmented = self.transform(image=image)
        img_tensor = augmented["image"]  # (C, H, W)
        return img_tensor, str(chip_path), trans_tuple, crs_str, True


def _read_chip_header(chip_path):
    try:
        with rasterio.open(chip_path) as src:
            t = src.transform
            return chip_path.name, (t.c, t.f, t.a, -t.e, src.width, src.height)
    except Exception:
        return chip_path.name, None


def infer_chip_grid(chip_files, num_threads=16):
    """
    Work out the regular grid the chips were cut on, from their origins.

    Returns:
        Dict with "cells" (chip file name -> (col, row)), "chip_px" and "step_px",
        or None if the chips do not overlap or are not on a regular grid.
    """
    # suppress_stderr swaps a process-wide file descriptor, so it wraps the whole
    # pool rather than each thread
    with suppress_stderr(), ThreadPoolExecutor(max_workers=num_threads) as pool:
        headers = dict(pool.map(_read_chip_header, chip_files))
    headers = {k: v for k, v in headers.items() if v is not None}
    if len(headers) < 2:
        return None

    xs, ys, res_x, res_y, widths, heights = (
        np.array(v) for v in zip(*headers.values())
    )
    if not (
        np.allclose(res_x, res_x[0])
        and np.allclose(res_y, res_x[0])
        and np.all(widths == widths[0])
        and np.all(heights == widths[0])
    ):
        logging.warning(
            "Chips differ in size or resolution; overlaps will not be resolved."
        )
        return None
    res = res_x[0]
    chip_px = int(widths[0])

    # smallest spacing between distinct chip origins, in pixels
    steps = [
        np.diff(np.unique(np.round((v - v.min()) / res).astype(int))) for v in (xs, ys)
    ]
    steps = np.concatenate([s for s in steps if len(s)])
    if not len(steps):
        return None
    step_px = int(steps.min())
    if step_px >= chip_px:
        return None

    cols = np.round((xs - xs.min()) / (step_px * res)).astype(int)
    rows = np.round((ys.max() - ys) / (step_px * res)).astype(int)
    on_grid = np.allclose(xs, xs.min() + cols * step_px * res, atol=res / 2) and (
        np.allclose(ys, ys.max() - rows * step_px * res, atol=res / 2)
    )
    if not on_grid:
        logging.warning(
            "Chip origins are not on a regular grid; overlaps will not be resolved."
        )
        return None

    logging.info(
        f"Chip grid: {chip_px} px chips, {step_px} px step, "
        f"{chip_px - step_px} px overlap."
    )
    cells = dict(zip(headers, zip(cols.tolist(), rows.tolist())))
    return {"cells": cells, "chip_px": chip_px, "step_px": step_px}


def compute_crop_margins(grid):
    """
    Work out how many pixels to trim from each side of each chip prediction so
    that overlapping chips tile the mosaic without overlap.

    Half the overlap is trimmed from every side that has a neighbouring chip, so
    each mosaic pixel comes from the chip where it sits furthest from the edge.
    Sides without a neighbour (edge of the chip set, or chips removed by the
    prediction mask) are kept whole. Where two such sides meet next to a missing
    chip they can overlap in a small corner, but only inside the missing chip's
    footprint, which is outside the prediction mask.

    Returns:
        Dict of chip file name -> (left, top, right, bottom) margins in pixels.
        Empty if grid is None.
    """
    if grid is None:
        return {}
    overlap_px = grid["chip_px"] - grid["step_px"]
    first = overlap_px // 2
    second = overlap_px - first

    occupied = set(grid["cells"].values())
    margins = {}
    for name, (c, r) in grid["cells"].items():
        margins[name] = (
            first if (c - 1, r) in occupied else 0,
            first if (c, r - 1) in occupied else 0,
            second if (c + 1, r) in occupied else 0,
            second if (c, r + 1) in occupied else 0,
        )
    return margins


def _writer_worker(write_queue, temp_dir, crop_margins):
    """
    Background thread worker that consumes (prob_map, chip_path, ...) items from
    write_queue and writes them as single-band float32 GeoTIFF files, decoupling
    disk I/O from GPU inference. Each prediction is cropped by its margins in
    crop_margins (if any) so the written tiles do not overlap.
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

        left, top, right, bottom = crop_margins.get(chip_path.name, (0, 0, 0, 0))
        if left or top or right or bottom:
            h, w = prob_map.shape
            prob_map = prob_map[top : h - bottom, left : w - right]
            transform = transform * Affine.translation(left, top)

        # nodata -1 rather than 0: fp16 sigmoid underflows to exactly 0 for
        # confident background, which must stay valid, while the intermediate VRTs
        # in build_vrt need a nodata value so their empty areas stay transparent
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "height": prob_map.shape[0],
            "width": prob_map.shape[1],
            "transform": transform,
            "crs": crs_str,
            "compress": "lzw",
            "nodata": -1,
        }

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(prob_map, 1)

        write_queue.task_done()


def predict_chips(
    model,
    input_dir,
    temp_dir,
    device,
    batch_size=32,
    num_workers=4,
    use_tta=False,
    overlap_mode="crop",
):
    """
    Run batched inference on all chips in input_dir and save to temp_dir.

    Args:
        batch_size: Number of chips per GPU forward pass.
        num_workers: CPU workers for DataLoader prefetch.
        overlap_mode: How overlapping chip predictions are resolved. "crop" trims
            half the overlap from each side so the tiles do not overlap; "none"
            writes every chip's full prediction, for per-chip diagnostics.
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

    grid = infer_chip_grid(chip_files) if overlap_mode == "crop" else None
    crop_margins = compute_crop_margins(grid)

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
            target=_writer_worker,
            args=(write_queue, temp_dir, crop_margins),
            daemon=True,
        )
        t.start()
        writer_threads.append(t)

    output_files = []
    unreadable = []
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
                else:
                    unreadable.append(Path(chip_path_str).name)

    # Signal writer to finish and wait for all writes to complete
    write_queue.put(None)
    for t in writer_threads:
        t.join()

    if unreadable:
        # neighbouring chips are cropped towards these, so their area is left
        # with no prediction at all
        logging.warning(
            f"{len(unreadable)} chips could not be read and have no prediction; "
            f"their area will have no lines: {', '.join(unreadable[:10])}"
            + (" ..." if len(unreadable) > 10 else "")
        )

    return output_files


def _build_vrt_file(vrt_path, input_files):
    """
    Build one VRT and raise if GDAL fails. Without GDAL exceptions turned on,
    BuildVRT returns None on a total failure, and an input it cannot read is
    skipped with only an error message, leaving a hole in the mosaic. Both are
    caught here by recording GDAL errors during the build.
    """
    errors = []

    def record_errors(err_class, err_no, msg):
        if err_class >= gdal.CE_Failure:
            errors.append(msg)

    options = gdal.BuildVRTOptions(
        resampleAlg=gdal.GRA_NearestNeighbour, resolution="highest"
    )
    gdal.PushErrorHandler(record_errors)
    try:
        ds = gdal.BuildVRT(
            str(vrt_path), [str(f) for f in input_files], options=options
        )
        built = ds is not None
        # closing writes the VRT to disk, so a write failure is caught here too
        ds = None
    finally:
        gdal.PopErrorHandler()
    if not built or errors:
        detail = "; ".join(errors[:5]) or gdal.GetLastErrorMsg()
        raise RuntimeError(f"GDAL could not build {vrt_path} completely: {detail}")


def _build_vrt_worker(args):
    vrt_path, input_files = args
    try:
        _build_vrt_file(vrt_path, input_files)
        return str(vrt_path)
    except Exception as e:
        logging.error(f"Worker VRT error: {e}")
        return None


def build_vrt(vrt_path, input_files):
    """Build a VRT from a list of input files. Returns False if any part fails."""
    logging.info(f"Building VRT from {len(input_files)} files...")

    if len(input_files) < 2000:
        try:
            _build_vrt_file(vrt_path, input_files)
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

        # a missing chunk would leave a hole in the mosaic and silently no lines there
        if len(intermediate_vrts) != len(worker_args):
            logging.error(
                f"{len(worker_args) - len(intermediate_vrts)} of {len(worker_args)} "
                "intermediate VRTs failed; not building an incomplete mosaic."
            )
            return False

        logging.info("Merging intermediate VRTs into final mosaic...")
        _build_vrt_file(vrt_path, intermediate_vrts)

        return True
    except Exception as e:
        logging.error(f"Error building VRT: {e}")
        return False


# the 8 neighbours in order round the ring, as (row, col) offsets
_RING_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))


def crossing_number(skeleton):
    """
    For every pixel, count the separate runs of skeleton pixels round its 8-pixel
    ring (the number of background -> skeleton steps going round once).
    """
    padded = np.pad(skeleton.astype(bool), 1)
    h, w = skeleton.shape
    ring = [
        padded[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w] for dr, dc in _RING_OFFSETS
    ]
    runs = np.zeros((h, w), dtype=np.uint8)
    for previous, current in zip(ring[-1:] + ring[:-1], ring):
        runs += current & ~previous
    return runs


# orthogonal neighbours first, so a walk along a staircase does not cut a corner
# and strand the corner pixel
_NEIGHBOUR_OFFSETS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


def trace_skeleton_paths(skeleton):
    """
    Trace a junction-free 1-pixel skeleton into ordered pixel paths, visiting each
    pixel once. Open lines are walked from one end to the other; closed loops are
    returned with the first pixel repeated at the end.

    Returns:
        List of (N, 2) int arrays of (row, col) pixel indices.
    """
    rows, cols = np.nonzero(skeleton)
    pixels = set(zip(rows.tolist(), cols.tolist()))

    def neighbours(p):
        return [
            (p[0] + dr, p[1] + dc)
            for dr, dc in _NEIGHBOUR_OFFSETS
            if (p[0] + dr, p[1] + dc) in pixels
        ]

    visited = set()

    def walk(start):
        path = [start]
        visited.add(start)
        current = start
        while True:
            nxt = next((q for q in neighbours(current) if q not in visited), None)
            if nxt is None:
                return path
            visited.add(nxt)
            path.append(nxt)
            current = nxt

    paths = []
    # ends of open lines first, then whatever is left (closed loops). An end is a
    # pixel with one run of neighbours round it, not one neighbour: a line can end
    # on a staircase step, where the last pixel touches two neighbours that also
    # touch each other. Counting neighbours would miss that end and start the walk
    # mid-line, splitting the line in two.
    runs = crossing_number(skeleton)
    ends = sorted(p for p in pixels if runs[p] <= 1)
    for start in ends + sorted(pixels):
        if start in visited:
            continue
        path = walk(start)
        last = path[-1]
        if len(path) > 2 and max(abs(last[0] - start[0]), abs(last[1] - start[1])) == 1:
            path.append(start)
        paths.append(np.array(path))
    return paths


def _drop_collinear(coords):
    """Remove vertices that sit on a straight run between their neighbours."""
    if len(coords) < 3:
        return coords
    steps = np.diff(coords, axis=0)
    keep = np.ones(len(coords), dtype=bool)
    keep[1:-1] = np.any(steps[1:] != steps[:-1], axis=1)
    return coords[keep]


def _split_to_core(path, core_cols, core_rows):
    """
    Keep the parts of a pixel path (global (col, row) indices) that belong to this
    chunk's core. A segment belongs to the chunk whose core holds its
    lexicographically smaller end pixel, so every segment is kept by exactly one
    chunk, and pieces from neighbouring chunks share their end pixels exactly.
    """
    seg_start, seg_end = path[:-1], path[1:]
    first_is_smaller = (seg_start[:, 0] < seg_end[:, 0]) | (
        (seg_start[:, 0] == seg_end[:, 0]) & (seg_start[:, 1] <= seg_end[:, 1])
    )
    owner = np.where(first_is_smaller[:, None], seg_start, seg_end)
    owned = (
        (owner[:, 0] >= core_cols[0])
        & (owner[:, 0] < core_cols[1])
        & (owner[:, 1] >= core_rows[0])
        & (owner[:, 1] < core_rows[1])
    )
    if owned.all():
        return [path]

    pieces = []
    # runs of consecutive owned segments; segment i joins vertices i and i+1
    padded = np.concatenate([[False], owned, [False]]).astype(np.int8)
    run_starts = np.flatnonzero(np.diff(padded) == 1)
    run_ends = np.flatnonzero(np.diff(padded) == -1)
    for s, e in zip(run_starts, run_ends):
        pieces.append(path[s : e + 1])
    return pieces


def process_chunk_worker(args):
    """
    Multiprocessing worker that reads one spatial chunk of the prediction VRT,
    plus a halo of neighbouring pixels, applies morphological closing,
    skeletonises the binary mask, removes junction pixels, traces the skeleton
    into lines, and returns the line coordinate arrays that fall in the chunk.

    The halo means the skeleton near the chunk edge is the same as the one the
    neighbouring chunk sees, so lines can be joined back up across chunks.
    """
    (
        vrt_path,
        col_start,
        row_start,
        width,
        height,
        chunk_size,
        halo,
        threshold,
        min_contour_length,
        transform_tuple,
    ) = args
    transform = Affine(*transform_tuple)

    with suppress_stderr():
        with rasterio.open(vrt_path) as src:
            chunk_height = min(chunk_size, height - row_start)
            chunk_width = min(chunk_size, width - col_start)
            read_col = max(0, col_start - halo)
            read_row = max(0, row_start - halo)
            read_width = min(width, col_start + chunk_width + halo) - read_col
            read_height = min(height, row_start + chunk_height + halo) - read_row
            window = Window(read_col, read_row, read_width, read_height)
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

            # 3. Junction breaker: remove branch points, i.e. skeleton pixels with
            # 3+ separate runs of skeleton pixels around their 8-pixel ring.
            # Counting neighbours instead would also break single lines at
            # staircase steps, where a pixel has 3 neighbours in 2 runs.
            # Repeat until none are left: removing a branch point can leave the
            # pixels round it touching diagonally, still joining the branches.
            while True:
                junctions = (crossing_number(skeleton_chunk) >= 3) & (
                    skeleton_chunk == 1
                )
                if not junctions.any():
                    break
                skeleton_chunk[junctions] = 0

            # 4. Trace each skeleton line once (cv2.findContours would trace around
            # the 1-pixel line, giving an out-and-back path of double the length)
            core_cols = (col_start, col_start + chunk_width)
            core_rows = (row_start, row_start + chunk_height)
            chunk_lines = []
            for path in trace_skeleton_paths(skeleton_chunk):
                if len(path) < max(min_contour_length, 2):
                    continue
                # (row, col) in the window -> global (col, row)
                path_global = np.column_stack(
                    (path[:, 1] + read_col, path[:, 0] + read_row)
                )
                for piece in _split_to_core(path_global, core_cols, core_rows):
                    piece = _drop_collinear(piece).astype(float)
                    xs_map, ys_map = transform * (piece[:, 0] + 0.5, piece[:, 1] + 0.5)
                    chunk_lines.append(np.column_stack((xs_map, ys_map)))

            return chunk_lines


def process_vrt_to_lines(
    vrt_path, chunk_size=2048, threshold=0.5, min_contour_length=5, halo=64
):
    """
    Process VRT in chunks to create skeleton and vectorize. Chunks overlap by
    `halo` pixels; line pieces cut at chunk edges are merged back together.
    """
    logging.info("Opening VRT for chunked processing...")

    with rasterio.open(vrt_path) as src:
        width = src.width
        height = src.height
        transform = src.transform
        crs = src.crs

        logging.info(f"Mosaic dimensions: {width} x {height} pixels")

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
                        halo,
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

        # join pieces that were cut at chunk edges; pieces only share end
        # vertices where a line crosses a chunk edge
        if lines:
            n_pieces = len(lines)
            merged = shapely.line_merge(MultiLineString(lines))
            lines = list(getattr(merged, "geoms", [merged]))
            logging.info(f"Merged {n_pieces} line pieces into {len(lines)} lines.")

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


def parse_arguments(args=None):
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
        help="Minimum number of skeleton pixels a predicted line segment must have to be kept. Shorter segments are discarded. Default: 5.",
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
        "--overlap-mode",
        choices=["crop", "none"],
        default="crop",
        help="How overlapping chip predictions are combined. 'crop' keeps each pixel "
        "from the chip where it is furthest from the edge; use it for production "
        "lines. 'none' keeps every chip's full "
        "prediction in temp_preds (for per-chip diagnostics such as example_plots.py); "
        "overlaps in the mosaic then come from whichever chip is listed last. "
        "Default: crop.",
    )
    parser.add_argument(
        "--prediction-mask",
        type=Path,
        default=None,
        help="Path to the prediction mask file (.gpkg or .shp). Output prediction lines are "
        "clipped to the mask before saving.",
    )
    parser.add_argument(
        "--output-gpkg",
        type=Path,
        default=None,
        help="Exact path for the output boundaries GeoPackage. Default: "
        "<output-dir>/<timestamp>_<model>_boundaries.gpkg. Use this when another "
        "step needs to know the output path in advance.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="If <output-dir>/temp_preds already holds chip predictions, skip "
        "inference and build the lines from them (e.g. to re-run vectorising with "
        "a different threshold). Without it, existing predictions are overwritten.",
    )

    return parser.parse_args(args)


def main(args):
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
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = args.output_dir / "temp_preds"

    use_existing = False
    if temp_dir.exists() and any(temp_dir.iterdir()):
        if args.reuse_existing:
            use_existing = True
            logging.info(
                f"Skipping inference, using existing predictions in {temp_dir}"
            )
        else:
            logging.info(f"Overwriting existing predictions in {temp_dir}")
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
            overlap_mode=args.overlap_mode,
        )

    if not pred_files:
        logging.warning("No predictions generated. Exiting.")
        sys.exit(1)

    # 4. Stitch (VRT)
    vrt_path = args.output_dir / "mosaic.vrt"
    if not build_vrt(vrt_path, pred_files):
        sys.exit(1)

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
            # clip returns MultiLineStrings; explode back to single lines
            gdf_lines = gpd.clip(gdf_lines, mask_union).explode(index_parts=False)
            lines = list(gdf_lines.geometry)
            logging.info(
                f"Clipped prediction lines to mask: {len(lines)} lines remaining."
            )
            if not lines:
                logging.warning("All lines were outside the prediction mask.")

    # 8. Save Output
    if lines:
        if args.output_gpkg is not None:
            out_gpkg = args.output_gpkg
            out_gpkg.parent.mkdir(parents=True, exist_ok=True)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = args.model.stem
            out_gpkg = args.output_dir / f"{timestamp}_{model_name}_boundaries.gpkg"

        logging.info(f"Saving {len(lines)} boundaries to {out_gpkg}...")
        gdf = gpd.GeoDataFrame(geometry=lines, crs=crs)
        gdf.to_file(out_gpkg, driver="GPKG")
        logging.info("Done.")
    else:
        logging.error("No lines detected.")

    # 9. Cleanup
    if not args.keep_preds:
        logging.info("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
        if vrt_path.exists():
            vrt_path.unlink()
    else:
        logging.info(f"Temporary files retained in {temp_dir}")

    if not lines:
        sys.exit(1)


if __name__ == "__main__":
    main(parse_arguments())
