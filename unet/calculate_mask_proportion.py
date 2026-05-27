import argparse
import random
from pathlib import Path
import rasterio
import numpy as np
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Calculate the mean proportion of '1' pixels in a sample of mask chips."
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        required=True,
        help="Directory containing the mask TIFF files.",
    )
    parser.add_argument(
        "--percent",
        type=float,
        default=10.0,
        help="Percentage of masks to sample (0-100). Default is 10.0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if not args.mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {args.mask_dir}")
    if not (0 < args.percent <= 100):
        raise ValueError(
            "Percent must be greater than 0 and less than or equal to 100."
        )

    mask_files = list(args.mask_dir.glob("*.tif"))
    if not mask_files:
        print(f"No .tif files found in {args.mask_dir}")
        return

    num_samples = max(1, int(len(mask_files) * (args.percent / 100.0)))

    random.seed(args.seed)
    sampled_files = random.sample(mask_files, num_samples)

    print(f"Sampling {num_samples} out of {len(mask_files)} masks ({args.percent}%)...")
    proportions = []

    for mask_path in tqdm(sampled_files, desc="Processing masks"):
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            proportions.append(np.count_nonzero(mask == 1) / mask.size)

    mean_proportion = np.mean(proportions)
    print(
        f"\nMean proportion across sample: {mean_proportion:.6f} ({mean_proportion * 100:.4f}%)"
    )


if __name__ == "__main__":
    main()
