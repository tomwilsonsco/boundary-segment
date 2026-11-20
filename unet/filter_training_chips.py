"""Identify and record low-quality training chips in chips_ignore.csv based on per-chip metrics."""

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_removal_condition(row, min_length, recall_min, min_precision):
    """Evaluates a row against the thresholds and returns a pipe-separated string of reasons."""
    reasons = []

    # corner-clippers
    if (row["Total_True_Length"] > 0) and (row["Total_True_Length"] < min_length):
        reasons.append("corner_clipper")

    # invisible boundaries (high FN / low recall)
    if (row["Total_True_Length"] >= 20) and (row["recall"] < recall_min):
        reasons.append("missed_or_invisible_boundary")

    # extra boundary / incorrect ground truth (high FP / low precision)
    if (row["FP_length"] >= 20) and (row["precision"] < min_precision):
        reasons.append("extra_boundary")

    return " | ".join(reasons) if reasons else None


def sample_cap_removals(chip_names, ignored, mask_dir, max_chips, seed):
    """Randomly cap the chips left for the dataset split, after filtering.

    The pool follows the split's own rules: chips not in the ignore list and not
    background-only in mask_dir/background_only_check.csv. A seeded random sample
    of max_chips is kept; the rest are returned for the ignore list.

    Args:
        chip_names:  All chip file names (the chips index).
        ignored:     File names already in the ignore list.
        mask_dir:    Directory holding background_only_check.csv.
        max_chips:   Number of chips to keep for the dataset split.
        seed:        Random seed, so the same inputs give the same sample.

    Returns:
        DataFrame of file_name / remove_condition rows for the capped-out chips.
    """
    bg_csv = mask_dir / "background_only_check.csv"
    if not bg_csv.exists():
        raise FileNotFoundError(
            f"{bg_csv} not found. It is written by split_dataset_train_test.py and "
            "is needed to count the chips the split will use."
        )
    bg_df = pd.read_csv(bg_csv)
    is_bg = bg_df["is_background_only"].astype(str).str.strip().str.lower()
    background = {
        Path(f).name for f in bg_df.loc[is_bg.isin(["true", "1", "yes"]), "mask_file"]
    }

    pool = sorted(set(chip_names) - set(ignored) - background)
    logging.info(f"Chips eligible for the dataset after filtering: {len(pool)}")
    if len(pool) <= max_chips:
        logging.info(f"Within the cap of {max_chips}, so no chips were sampled out.")
        return pd.DataFrame(columns=["file_name", "remove_condition"])

    rng = np.random.default_rng(seed)
    keep = set(rng.choice(pool, size=max_chips, replace=False))
    capped = [name for name in pool if name not in keep]
    logging.info(
        f"Randomly kept {max_chips} chips (seed {seed}); "
        f"{len(capped)} recorded as 'sample_cap'."
    )
    return pd.DataFrame({"file_name": capped, "remove_condition": "sample_cap"})


def parse_arguments(args=None):
    """Set up and parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter noisy training chips based on evaluation metrics."
    )
    parser.add_argument(
        "--input-gpkg",
        type=Path,
        required=True,
        help="Path to the chip metrics GeoPackage produced by chip_metrics.py. Typically named chips_index_metrics.gpkg.",
    )
    parser.add_argument(
        "--chip-dir",
        type=Path,
        required=True,
        help="Path to the chips directory. chips_ignore.csv will be created here (or updated if it already exists) with the filtered chip names.",
    )
    parser.add_argument(
        "--min-training-length",
        type=float,
        default=30.0,
        help="Minimum total true boundary length in metres. Chips below this are 'corner_clippers'. Default: 30.0",
    )
    parser.add_argument(
        "--recall-min",
        type=float,
        default=0.5,
        help="Minimum recall threshold. If recall is below this (and true length >= 20m), it is recorded as 'missed_or_invisible_boundary'. Default: 0.5",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.5,
        help="Minimum precision threshold. If precision is below this (and FP length >= 20m), it is recorded as 'extra_boundary'. Default: 0.5",
    )
    parser.add_argument(
        "--max-dataset-chips",
        type=int,
        default=None,
        help="Cap on the chips passed to the dataset split (train+val+test), applied "
        "after filtering as a seeded random sample. Chips sampled out are recorded "
        "as 'sample_cap'. Requires --mask-dir. Default: no cap.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Directory containing background_only_check.csv (written by "
        "split_dataset_train_test.py). Needed with --max-dataset-chips.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the --max-dataset-chips sample. Default: 42",
    )
    return parser.parse_args(args)


def main(args):
    """Main orchestration function."""
    input_path = args.input_gpkg
    chip_dir = args.chip_dir

    if args.max_dataset_chips is not None:
        if args.max_dataset_chips < 1:
            raise ValueError("--max-dataset-chips must be at least 1.")
        if args.mask_dir is None:
            raise ValueError("--max-dataset-chips needs --mask-dir.")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logging.info(f"Loading metrics from {input_path}...")
    gdf = gpd.read_file(input_path)

    # ensure required columns exist and fill any NaNs (e.g. 0/0 division in precision/recall)
    cols_to_check = ["TP_length", "FP_length", "FN_length", "precision", "recall"]
    for col in cols_to_check:
        if col not in gdf.columns:
            raise ValueError(f"Missing required column in input data: {col}")
        gdf[col] = pd.to_numeric(gdf[col], errors="coerce").fillna(0)

    # calculate total ground truth boundary length
    gdf["Total_True_Length"] = gdf["TP_length"] + gdf["FN_length"]

    logging.info("Evaluating chips against thresholds...")

    # Apply the logic row by row
    gdf["remove_condition"] = gdf.apply(
        lambda row: get_removal_condition(
            row,
            args.min_training_length,
            args.recall_min,
            args.min_precision,
        ),
        axis=1,
    )

    # filter to only the chips that triggered a condition
    to_remove = gdf[gdf["remove_condition"].notnull()].copy()

    if to_remove.empty:
        logging.info("No chips met the metrics criteria for removal.")
        new_removals_df = pd.DataFrame(columns=["file_name", "remove_condition"])
    else:
        if "file_name" not in to_remove.columns:
            raise ValueError("The column 'file_name' was not found in the input data.")
        new_removals_df = to_remove[["file_name", "remove_condition"]]

    csv_path = chip_dir / "chips_ignore.csv"

    # handle existing chips_ignore.csv - e.g. background only chips
    if csv_path.exists():
        # a read failure must stop the run: carrying on would drop the
        # outside-mask exclusions and let those chips into training
        existing_df = pd.read_csv(csv_path)
        conditions_to_keep = [
            "outside image bounds",
            "outside training",
            "outside prediction mask",
        ]
        if "remove_condition" in existing_df.columns:
            kept_df = existing_df[
                existing_df["remove_condition"].isin(conditions_to_keep)
            ]
        else:
            kept_df = pd.DataFrame(columns=["file_name", "remove_condition"])
        logging.info(
            f"Loaded existing list and kept {len(kept_df)} manually/externally defined exclusions."
        )
    else:
        kept_df = pd.DataFrame(columns=["file_name", "remove_condition"])

    # combine kept rows with new removals
    final_df = pd.concat([kept_df, new_removals_df], ignore_index=True)
    # keep the external conditions first if there's overlap
    final_df = final_df.drop_duplicates(subset=["file_name"], keep="first")

    # cap the chips left after filtering with a random sample, so large areas
    # train on a representative subset of the chips that passed
    if args.max_dataset_chips is not None:
        capped_df = sample_cap_removals(
            gdf["file_name"],
            final_df["file_name"],
            args.mask_dir,
            args.max_dataset_chips,
            args.seed,
        )
        final_df = pd.concat([final_df, capped_df], ignore_index=True)

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    final_df.to_csv(csv_path, index=False)

    # summary stats
    logging.info(
        f"Successfully identified {len(new_removals_df)} chips for removal based on metrics."
    )
    logging.info(f"Total chips in ignore list (including preserved): {len(final_df)}")

    if not final_df.empty:
        logging.info("Breakdown of removal conditions:")
        condition_counts = (
            final_df["remove_condition"]
            .astype(str)
            .str.split(" | ", regex=False)
            .explode()
            .value_counts()
        )
        for cond, count in condition_counts.items():
            logging.info(f"  - {cond}: {count}")

    logging.info(f"List saved to {csv_path}")


if __name__ == "__main__":
    main(parse_arguments())
