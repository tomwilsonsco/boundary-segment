"""Split chip images and masks into train, validation, and test dataset directories."""

import argparse
from pathlib import Path
from rschip import DatasetSplitter
import numpy as np
import pandas as pd
import logging

# set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_arguments(args=None):
    """Set up and parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Split an image and mask dataset into train, validation, and test sets."
    )
    # required arguments
    parser.add_argument(
        "--chip-dir",
        type=Path,
        required=True,
        help="Path to the directory containing all image files.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        required=True,
        help="Path to the directory containing all mask files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to the root directory where the split dataset will be created.",
    )
    # optional arguments with defaults
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Proportion of the dataset for training (default: 0.7).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Proportion of the dataset for validation (default: 0.2).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Proportion of the dataset for testing (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42).",
    )
    parser.add_argument(
        "--no-filter-background-only",
        dest="filter_background_only",
        action="store_false",
        help="If set, does NOT filter out image/mask pairs that only contain background.",
    )
    return parser.parse_args(args)


def main(args):
    """Main orchestration function."""
    # validate that the ratios sum to 1
    if not np.isclose(args.train_ratio + args.val_ratio + args.test_ratio, 1.0):
        raise ValueError("The sum of train, val, and test ratios must be 1.0.")
    if any(r < 0 for r in [args.train_ratio, args.val_ratio, args.test_ratio]):
        raise ValueError("Ratios cannot be negative.")

    # Check for and load an ignore list if it exists
    ignore_list = []
    ignore_csv_path = args.chip_dir / "chips_ignore.csv"
    if ignore_csv_path.exists():
        logging.info(f"Found ignore list at: {ignore_csv_path}")
        try:
            ignore_df = pd.read_csv(ignore_csv_path)
            if "file_name" in ignore_df.columns:
                ignore_list = ignore_df["file_name"].tolist()
                logging.info(f"Excluding {len(ignore_list)} chips from the dataset.")
            else:
                logging.warning(
                    "chips_ignore.csv found, but 'file_name' column is missing. No chips will be excluded."
                )
        except Exception as e:
            logging.error(f"Failed to read or process {ignore_csv_path}: {e}")

    splitter = DatasetSplitter(
        image_dir=args.chip_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        filter_background_only=args.filter_background_only,
        exclude_files=ignore_list,
    )

    splitter.split()


if __name__ == "__main__":
    parsed_args = parse_arguments()
    main(parsed_args)
