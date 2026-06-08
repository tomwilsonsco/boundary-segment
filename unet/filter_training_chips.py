import argparse
import logging
from pathlib import Path
import geopandas as gpd
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


def main():
    parser = argparse.ArgumentParser(
        description="Filter noisy training chips based on evaluation metrics."
    )
    parser.add_argument(
        "--input-gpkg",
        type=str,
        required=True,
        help="Path to the input chip metrics GeoPackage.",
    )
    parser.add_argument(
        "--chip-dir",
        type=str,
        required=True,
        help="Path to chips directory with chips_ignore.csv.",
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
        default=0.2,
        help="Minimum precision threshold. If precision is below this (and FP length >= 20m), it is recorded as 'extra_boundary'. Default: 0.2",
    )

    args = parser.parse_args()

    input_path = Path(args.input_gpkg)

    chips_dir = Path(args.chip_dir)

    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        return

    logging.info(f"Loading metrics from {input_path}...")
    try:
        gdf = gpd.read_file(input_path)
    except Exception as e:
        logging.error(f"Failed to read GeoPackage: {e}")
        return

    # ensure required columns exist and fill any NaNs (e.g. 0/0 division in precision/recall)
    cols_to_check = ["TP_length", "FP_length", "FN_length", "precision", "recall"]
    for col in cols_to_check:
        if col not in gdf.columns:
            logging.error(f"Missing required column in input data: {col}")
            return
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
            logging.error("The column 'file_name' was not found in the input data.")
            return
        new_removals_df = to_remove[["file_name", "remove_condition"]]

    csv_path = chips_dir / "chips_ignore.csv"

    # handle existing chips_ignore.csv - e.g. background only chips
    if csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path)
            conditions_to_keep = ["outside image bounds", "outside training"]
            if "remove_condition" in existing_df.columns:
                kept_df = existing_df[
                    existing_df["remove_condition"].isin(conditions_to_keep)
                ]
            else:
                kept_df = pd.DataFrame(columns=["file_name", "remove_condition"])
            logging.info(
                f"Loaded existing list and kept {len(kept_df)} manually/externally defined exclusions."
            )
        except Exception as e:
            logging.error(f"Failed to read existing CSV: {e}")
            kept_df = pd.DataFrame(columns=["file_name", "remove_condition"])
    else:
        kept_df = pd.DataFrame(columns=["file_name", "remove_condition"])

    # combine kept rows with new removals
    final_df = pd.concat([kept_df, new_removals_df], ignore_index=True)
    # keep the external conditions first if there's overlap
    final_df = final_df.drop_duplicates(subset=["file_name"], keep="first")

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
    main()
