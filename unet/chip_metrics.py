import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def calculate_f1_from_mean_pr(df):
    """
    Calculates F1 score from the mean of precision and recall columns.
    This is a macro-average F1.
    """
    # nan skews the mean
    valid_df = df.dropna(subset=["precision", "recall"])
    if valid_df.empty:
        return 0.0, 0.0, 0.0

    mean_precision = valid_df["precision"].mean()
    mean_recall = valid_df["recall"].mean()

    if (mean_precision + mean_recall) == 0:
        f1 = 0.0
    else:
        f1 = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)

    return mean_precision, mean_recall, f1


def parse_arguments(args=None):
    """Set up and parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate F1 score for training chips based on line comparison."
    )

    parser.add_argument(
        "--line-comparison",
        type=Path,
        required=True,
        help="Path to the line comparison GPKG (output of line_evaluate).",
    )
    parser.add_argument(
        "--chips-index",
        type=Path,
        required=True,
        help="Path to the chips index GPKG.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        required=True,
        help="Directory containing chip mask files. Must also contain background_only_check.csv, "
        "which is created automatically by split_dataset_train_test.py (rschip.DatasetSplitter).",
    )
    parser.add_argument(
        "--output-gpkg",
        type=Path,
        default=None,
        help="Path to save the output GPKG. Defaults to the line comparison directory with _chips.gpkg extension.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Path to dataset directory. If provided, checks images/train, images/val, images/test for split.",
    )

    return parser.parse_args(args)


def main(args):
    if not args.line_comparison.exists():
        raise FileNotFoundError(
            f"Line comparison file not found: {args.line_comparison}"
        )

    if not args.chips_index.exists():
        raise FileNotFoundError(f"Chips index file not found: {args.chips_index}")

    csv_path = args.mask_dir / "background_only_check.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"background_only_check.csv not found in {args.mask_dir}"
        )

    logging.info(f"Loading line comparison from {args.line_comparison}...")
    lines_gdf = gpd.read_file(args.line_comparison)

    if "pred_result" not in lines_gdf.columns:
        raise ValueError(
            "Input line comparison GPKG must contain a 'pred_result' column."
        )

    logging.info(f"Loading chips index from {args.chips_index}...")
    index_gdf = gpd.read_file(args.chips_index)

    if "file_name" not in index_gdf.columns:
        raise ValueError("Chips index GPKG must contain a 'file_name' column.")

    logging.info(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    if "is_background_only" not in df.columns:
        raise ValueError("CSV must contain 'is_background_only' column.")
    if "image_file" not in df.columns:
        raise ValueError("CSV must contain 'image_file' column.")

    # file name for joining
    df["file_name"] = df["image_file"].apply(lambda x: Path(x).name)

    logging.info("Joining index layer to CSV data...")
    gdf = index_gdf.merge(df, on="file_name", how="left")

    if args.dataset_dir:
        if not args.dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {args.dataset_dir}")
        if not (args.dataset_dir / "images" / "train").exists():
            raise ValueError(
                f"Invalid dataset directory (missing 'images/train'): {args.dataset_dir}"
            )

        split_map = {}
        for split in ["train", "val", "test"]:
            split_dir = args.dataset_dir / "images" / split
            if split_dir.exists():
                for f in split_dir.glob("*.tif"):
                    split_map[f.name] = split

        gdf["dataset_split"] = gdf["file_name"].map(split_map).fillna("NA")
    else:
        gdf["dataset_split"] = "NA"

    is_bg = (
        gdf["is_background_only"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )
    to_process = gdf[~is_bg]

    lines_sindex = lines_gdf.sindex
    results = []

    if to_process.empty:
        logging.info("No chips to process metrics for (all background).")
    else:
        logging.info(f"Intersecting lines with {len(to_process)} chip boundaries...")
        for _, row in tqdm(
            to_process.iterrows(), total=len(to_process), desc="Calculating metrics"
        ):
            geom = row["geometry"]
            file_name = row["file_name"]

            tp_len = 0.0
            fp_len = 0.0
            fn_len = 0.0
            has_features = False

            possible_matches_idx = list(lines_sindex.intersection(geom.bounds))
            if possible_matches_idx:
                possible_matches = lines_gdf.iloc[possible_matches_idx]
                clipped = gpd.clip(possible_matches, geom)

                if not clipped.empty:
                    tp_len = clipped[
                        clipped["pred_result"] == "TP"
                    ].geometry.length.sum()
                    fp_len = clipped[
                        clipped["pred_result"] == "FP"
                    ].geometry.length.sum()
                    fn_len = clipped[
                        clipped["pred_result"] == "FN"
                    ].geometry.length.sum()

                    if (tp_len + fp_len + fn_len) > 0:
                        has_features = True

            if not has_features:
                gdf.loc[gdf["file_name"] == file_name, "is_background_only"] = True
                continue

            precision = tp_len / (tp_len + fp_len) if (tp_len + fp_len) > 0 else 0.0
            recall = tp_len / (tp_len + fn_len) if (tp_len + fn_len) > 0 else 0.0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            results.append(
                {
                    "file_name": file_name,
                    "TP_length": tp_len,
                    "FP_length": fp_len,
                    "FN_length": fn_len,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                }
            )

    metrics_cols = [
        "TP_length",
        "FP_length",
        "FN_length",
        "precision",
        "recall",
        "f1_score",
    ]
    if results:
        results_df = pd.DataFrame(results)
        for col in metrics_cols:
            if col in gdf.columns:
                gdf = gdf.drop(columns=[col])
        gdf = gdf.merge(results_df, on="file_name", how="left")
    else:
        for col in metrics_cols:
            if col not in gdf.columns:
                gdf[col] = np.nan

    out_gpkg = (
        args.output_gpkg
        if args.output_gpkg
        else args.line_comparison.parent / f"{args.line_comparison.stem}_chips.gpkg"
    )
    logging.info(f"Saving results to {out_gpkg}...")
    gdf.to_file(out_gpkg, driver="GPKG")
    logging.info("Done.")

    if results:
        logging.info("\n" + "=" * 40)
        logging.info("Mean Metrics per Chip")
        logging.info("=" * 40)

        logging.info("Overall:")
        p, r, f1 = calculate_f1_from_mean_pr(gdf)
        logging.info(f"  Precision: {p:.4f}")
        logging.info(f"  Recall:    {r:.4f}")
        logging.info(f"  F1 Score:  {f1:.4f}")

        if args.dataset_dir and "dataset_split" in gdf.columns:
            logging.info("\nBy Dataset Split:")
            splits = sorted(gdf["dataset_split"].unique())
            for split in splits:
                split_gdf = gdf[gdf["dataset_split"] == split]
                if not split_gdf.empty:
                    p, r, f1 = calculate_f1_from_mean_pr(split_gdf)
                    logging.info(f"  {split.upper()}:")
                    logging.info(f"    Precision: {p:.4f}")
                    logging.info(f"    Recall:    {r:.4f}")
                    logging.info(f"    F1 Score:  {f1:.4f}")
        logging.info("=" * 40 + "\n")


if __name__ == "__main__":
    parsed_args = parse_arguments()
    main(parsed_args)
