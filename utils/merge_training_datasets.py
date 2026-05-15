import argparse
import logging
import multiprocessing
import shutil
from functools import partial
from pathlib import Path
from tqdm import tqdm

# basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

SUBDIRS = [
    Path("images") / "train",
    Path("images") / "val",
    Path("images") / "test",
    Path("masks") / "train",
    Path("masks") / "val",
    Path("masks") / "test",
]

def rename_file(filepath: Path, prefix: str):
    """Worker function to rename a single file."""
    if not filepath.name.startswith(f"{prefix}_"):
        new_name = f"{prefix}_{filepath.name}"
        new_filepath = filepath.parent / new_name
        filepath.rename(new_filepath)
        return True
    return False

def rename_and_prefix(dataset_dir: Path, num_workers: int):
    prefix = dataset_dir.parent.name
    logging.info(f"Prefixing files in {dataset_dir} with '{prefix}_'...")
    
    all_files = []
    for subdir in SUBDIRS:
        dir_path = dataset_dir / subdir
        if dir_path.exists():
            all_files.extend(list(dir_path.glob("*.tif")))

    count = 0
    rename_func = partial(rename_file, prefix=prefix)
    with multiprocessing.Pool(num_workers) as pool:
        for changed in tqdm(pool.imap_unordered(rename_func, all_files), total=len(all_files), desc=f"Renaming {prefix}"):
            if changed:
                count += 1
    logging.info(f"Renamed {count} files in {dataset_dir}.")

def move_file(args):
    """Worker function to move a single file."""
    filepath, target_filepath = args
    if target_filepath.exists():
        return False, f"File already exists in target: {target_filepath}. Skipping."
    shutil.move(str(filepath), str(target_filepath))
    return True, ""

def parse_arguments():
    parser = argparse.ArgumentParser(description="Merge two training datasets.")
    parser.add_argument(
        "--target-dataset", type=Path, required=True, help="Path to the target dataset directory."
    )
    parser.add_argument(
        "--source-dataset", type=Path, required=True, help="Path to the source dataset directory."
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1), help="Number of worker processes."
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    target_dataset = args.target_dataset.resolve()
    source_dataset = args.source_dataset.resolve()

    if not target_dataset.exists():
        logging.error(f"Target dataset directory not found: {target_dataset}")
        return
    if not source_dataset.exists():
        logging.error(f"Source dataset directory not found: {source_dataset}")
        return

    logging.info("Starting merge process...")

    # rename files in both datasets to avoid name collisions
    rename_and_prefix(target_dataset, args.workers)
    rename_and_prefix(source_dataset, args.workers)

    # move files from source to target
    logging.info(f"Moving files from {source_dataset} to {target_dataset}...")
    move_tasks = []
    for subdir in SUBDIRS:
        source_dir = source_dataset / subdir
        target_dir = target_dataset / subdir

        if source_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            for filepath in source_dir.glob("*.tif"):
                target_filepath = target_dir / filepath.name
                move_tasks.append((filepath, target_filepath))

    moved_count = 0
    with multiprocessing.Pool(args.workers) as pool:
        for success, msg in tqdm(pool.imap_unordered(move_file, move_tasks), total=len(move_tasks), desc="Moving files"):
            if success:
                moved_count += 1
            elif msg:
                logging.warning(msg)

    logging.info(f"Moved {moved_count} files.")
    logging.info("Merge complete.")

if __name__ == "__main__":
    main()
