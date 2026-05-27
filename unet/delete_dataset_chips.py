import argparse
import logging
from pathlib import Path
import pandas as pd

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Delete chips from a train/val/test dataset that appear in chips_ignore.csv."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to the root dataset directory containing train/, val/, test/ splits.",
    )
    parser.add_argument(
        "--chips-dir",
        type=Path,
        required=True,
        help="Path to the chips directory containing chips_ignore.csv.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, log what would be deleted without actually deleting anything.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train"],
        help="Which splits to delete chips from. Can be any combination of train, val, test (e.g., --splits train test). Default: train.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    csv_path = args.chips_dir / "chips_ignore.csv"
    if not csv_path.exists():
        logging.error(f"chips_ignore.csv not found at: {csv_path}")
        return

    try:
        ignore_df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"Failed to read chips_ignore.csv: {e}")
        return

    if "file_name" not in ignore_df.columns:
        logging.error("chips_ignore.csv does not contain a 'file_name' column.")
        return

    file_names = ignore_df["file_name"].dropna().tolist()
    logging.info(f"Loaded {len(file_names)} entries from chips_ignore.csv.")

    if not args.dataset_dir.exists():
        logging.error(f"Dataset directory not found: {args.dataset_dir}")
        return

    deleted = 0
    not_found = 0

    for file_name in file_names:
        for split in args.splits:
            for subdir in ("images", "masks"):
                target = args.dataset_dir / subdir / split / file_name
                if target.exists():
                    if args.dry_run:
                        logging.info(f"[dry-run] Would delete: {target}")
                    else:
                        target.unlink()
                        logging.info(f"Deleted: {target}")
                    deleted += 1
                else:
                    not_found += 1

    if args.dry_run:
        logging.info(f"Dry run complete. Would have deleted {deleted} file(s).")
    else:
        logging.info(
            f"Done. Deleted {deleted} file(s). {not_found} paths were not found."
        )


if __name__ == "__main__":
    main()
