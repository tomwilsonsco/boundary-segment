"""Generate binary segmentation mask GeoTIFFs from land parcel polygon boundaries for each chip image.

A mask pixel is 1 where its centre lies within --buffer-dist of a parcel boundary.
Each chip only uses the boundaries near it, so memory does not grow with the size
of the area: each worker holds the parcel boundaries once and builds a chip's
buffer from the few lines that reach it.
"""

import argparse
import logging
import multiprocessing
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio as rio
import shapely
from rasterio.features import rasterize
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# set in each worker by init_worker, so the boundaries are loaded once per process
# rather than once per chip
_LINES = None
_TREE = None
_CRS = None
_BUFFER = None


def load_boundaries(parcels_path):
    """Parcel boundaries as single-part lines, with the parcels' CRS."""
    gdf = gpd.read_file(parcels_path)
    lines = gdf.geometry.boundary.explode(index_parts=False)
    lines = lines[~lines.is_empty]
    return lines.values.to_numpy(), gdf.crs


def init_worker(parcels_path, buffer_dist):
    """Load the parcel boundaries and a spatial index into this worker."""
    global _LINES, _TREE, _CRS, _BUFFER
    _LINES, _CRS = load_boundaries(parcels_path)
    _TREE = shapely.STRtree(_LINES)
    _BUFFER = buffer_dist


def chip_mask(lines, tree, transform, shape, bounds, buffer_dist):
    """Rasterise the buffered boundaries that reach one chip."""
    # lines are cut at a margin wide enough that the round ends left by the cut
    # stay outside the chip, so the buffer inside the chip is unchanged
    margin = 3 * buffer_dist + 1.0
    xmin, ymin, xmax, ymax = bounds
    rect = (xmin - margin, ymin - margin, xmax + margin, ymax + margin)
    idx = tree.query(shapely.box(*rect))
    if len(idx) == 0:
        return np.zeros(shape, dtype=np.uint8)
    near = shapely.clip_by_rect(lines[idx], *rect)
    near = near[~shapely.is_empty(near)]
    if len(near) == 0:
        return np.zeros(shape, dtype=np.uint8)
    # a pixel is burned if its centre falls in any buffer, so the overlapping
    # buffers of shared edges do not need to be dissolved first
    buffered = shapely.buffer(near, buffer_dist)
    return rasterize(
        ((geom, 1) for geom in buffered),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )


def process_mask_creation(chip_path, out_dir):
    """Worker function to create a single mask."""
    try:
        with rio.open(chip_path) as src:
            meta = src.meta.copy()
            transform, bounds, crs = src.transform, src.bounds, src.crs
            shape = (src.height, src.width)
        if _CRS is not None and crs is not None and _CRS != crs:
            raise ValueError(f"Chip CRS {crs} does not match the parcels CRS {_CRS}")
        mask = chip_mask(_LINES, _TREE, transform, shape, bounds, _BUFFER)
        # the same output format as rschip.SegmentationMask, which this replaces
        meta.update({"count": 1, "dtype": "uint8", "compress": "lzw", "nodata": None})
        with rio.open(out_dir / chip_path.name, "w", **meta) as dst:
            dst.write(mask, 1)
        return True
    except Exception as e:
        logging.error(f"Error processing {chip_path.name}: {e}")
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
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: available cores - 1)",
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

    chip_paths = sorted(chip_dir.glob("*.tif"))
    logging.info(f"Processing {len(chip_paths)} chips...")

    success_count = 0
    if not args.singleprocessor:
        num_workers = args.workers or max(1, multiprocessing.cpu_count() - 1)
        logging.info(f"Using {num_workers} workers for processing.")
        with multiprocessing.Pool(
            num_workers,
            initializer=init_worker,
            initargs=(parcels, args.buffer_dist),
        ) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(
                        _process_in_worker,
                        [(p, out_dir) for p in chip_paths],
                        chunksize=16,
                    ),
                    total=len(chip_paths),
                    desc="Generating masks",
                )
            )
        success_count = sum(results)
    else:
        logging.info("Using single process.")
        init_worker(parcels, args.buffer_dist)
        for chip_path in tqdm(chip_paths, desc="Generating masks"):
            if process_mask_creation(chip_path, out_dir):
                success_count += 1

    failed_count = len(chip_paths) - success_count
    logging.info(
        f"Mask generation complete. Succeeded: {success_count}, Failed: {failed_count}"
    )
    if failed_count:
        raise RuntimeError(f"{failed_count} masks could not be created")


def _process_in_worker(task):
    """Unpack a (chip_path, out_dir) task for Pool.imap_unordered."""
    return process_mask_creation(*task)


if __name__ == "__main__":
    parsed_args = parse_arguments()
    main(parsed_args)
