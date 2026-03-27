import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


def parse_arguments(args=None):
    """Set up and parse command line arguments."""
    parser = argparse.ArgumentParser(description="Swap mask chips in a dataset with new masks.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to the target dataset directory (containing masks/train, masks/val, masks/test).",
    )
    parser.add_argument(
        "--new-masks-dir",
        type=Path,
        required=True,
        help="Path to the directory containing new mask tifs.",
    )
    return parser.parse_args(args)


def main(args):
    dataset_dir = args.dataset_dir.resolve()
    new_masks_dir = args.new_masks_dir.resolve()

    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")
    if not new_masks_dir.exists():
        raise ValueError(f"New masks directory not found: {new_masks_dir}")

    mask_splits = ["train", "val", "test"]

    # First, collect all target masks and verify they exist in the new masks directory
    replacement_plan = []
    missing_masks = []

    for split in mask_splits:
        split_dir = dataset_dir / "masks" / split
        if not split_dir.exists():
            continue

        for mask_path in split_dir.glob("*.tif"):
            new_mask_path = new_masks_dir / mask_path.name
            if not new_mask_path.exists():
                missing_masks.append(mask_path.name)
            else:
                replacement_plan.append((new_mask_path, mask_path))

    if missing_masks:
        raise ValueError(
            f"Missing {len(missing_masks)} new masks in {new_masks_dir}. "
            f"Examples: {missing_masks[:5]}"
        )

    print(f"Found all {len(replacement_plan)} replacement masks. Proceeding with swap...")
    for src, dst in tqdm(replacement_plan, desc="Swapping masks"):
        shutil.copy2(src, dst)
    print(f"Successfully swapped {len(replacement_plan)} mask chips.")


if __name__ == "__main__":
    parsed_args = parse_arguments()
    main(parsed_args)