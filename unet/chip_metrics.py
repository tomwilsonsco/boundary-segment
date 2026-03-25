import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm


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
        help="Directory containing chip masks and background_only_check.csv.",
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

    print(f"Loading line comparison from {args.line_comparison}...")
    lines_gdf = gpd.read_file(args.line_comparison)

    if "pred_result" not in lines_gdf.columns:
        raise ValueError(
            "Input line comparison GPKG must contain a 'pred_result' column."
        )

    print(f"Loading chips index from {args.chips_index}...")
    index_gdf = gpd.read_file(args.chips_index)

    if "file_name" not in index_gdf.columns:
        raise ValueError("Chips index GPKG must contain a 'file_name' column.")

    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    if "is_background_only" not in df.columns:
        raise ValueError("CSV must contain 'is_background_only' column.")
    if "image_file" not in df.columns:
        raise ValueError("CSV must contain 'image_file' column.")

    # file name for joining
    df["file_name"] = df["image_file"].apply(lambda x: Path(x).name)

    print("Joining index layer to CSV data...")
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
        print("No chips to process metrics for (all background).")
    else:
        print(f"Intersecting lines with {len(to_process)} chip boundaries...")
        for _, row in tqdm(
            to_process.iterrows(), total=len(to_process), desc="Calculating metrics"
        ):
            geom = row["geometry"]
            file_name = row["file_name"]

            tp_len = 0.0
            fp_len = 0.0
            fn_len = 0.0

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
    print(f"Saving results to {out_gpkg}...")
    gdf.to_file(out_gpkg, driver="GPKG")
    print("Done.")

    if results:
        print("\n" + "=" * 40)
        print("Mean Metrics per Chip")
        print("=" * 40)

        valid_gdf = gdf.dropna(subset=["f1_score"])

        print("Overall:")
        print(f"  Precision: {valid_gdf['precision'].mean():.4f}")
        print(f"  Recall:    {valid_gdf['recall'].mean():.4f}")
        print(f"  F1 Score:  {valid_gdf['f1_score'].mean():.4f}")

        if args.dataset_dir:
            print("\nBy Dataset Split:")
            grouped = valid_gdf.groupby("dataset_split")[
                ["precision", "recall", "f1_score"]
            ].mean()
            for split, row in grouped.iterrows():
                print(f"  {split.upper()}:")
                print(f"    Precision: {row['precision']:.4f}")
                print(f"    Recall:    {row['recall']:.4f}")
                print(f"    F1 Score:  {row['f1_score']:.4f}")
        print("=" * 40 + "\n")


if __name__ == "__main__":
    parsed_args = parse_arguments()
    main(parsed_args)
