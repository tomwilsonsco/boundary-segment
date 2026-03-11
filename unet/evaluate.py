import os
from contextlib import contextmanager
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from datetime import datetime
import argparse
from pathlib import Path


@contextmanager
def suppress_stderr():
    """
    suppress C-level stderr output (like libtiff warnings).
    This redirects the file descriptor 2 (stderr) to /dev/null temporarily.
    """
    try:
        null_fd = os.open(os.devnull, os.O_RDWR)
        save_fd = os.dup(2)
        os.dup2(null_fd, 2)
        yield
    except Exception:
        yield
    finally:
        try:
            os.dup2(save_fd, 2)
            os.close(null_fd)
            os.close(save_fd)
        except Exception:
            pass


def get_preprocessing():
    """Get preprocessing transforms for inference."""
    return albu.Compose(
        [
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def calculate_iou(pred, target, threshold=0.5):
    """Calculate Intersection over Union (IoU) score."""
    pred_binary = (pred > threshold).astype(np.uint8)
    target_binary = (target > 0).astype(np.uint8)

    intersection = np.logical_and(pred_binary, target_binary).sum()
    union = np.logical_or(pred_binary, target_binary).sum()

    if union == 0:
        return 1.0

    return intersection / union


def calculate_dice(pred, target, threshold=0.5):
    """Calculate Dice coefficient (F1 score)."""
    pred_binary = (pred > threshold).astype(np.uint8)
    target_binary = (target > 0).astype(np.uint8)

    intersection = np.logical_and(pred_binary, target_binary).sum()

    pred_sum = pred_binary.sum()
    target_sum = target_binary.sum()

    if pred_sum + target_sum == 0:
        return 1.0 if intersection == 0 else 0.0

    return (2 * intersection) / (pred_sum + target_sum)


def load_model(model_path, device):
    """Load trained model from checkpoint."""
    # load file to check for metadata
    checkpoint = torch.load(model_path, map_location=device)

    state_dict = None
    arch_name = None
    encoder_name = None

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        print("Found model configuration in file.")
        state_dict = checkpoint["state_dict"]
        arch_name = checkpoint.get("arch")
        encoder_name = checkpoint.get("encoder")
    # or full training checkpoint
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        arch_name = checkpoint.get("arch")
        encoder_name = checkpoint.get("encoder")
    else:
        raise ValueError("Model checkpoint does not contain metadata (arch/encoder).")

    if arch_name is None or encoder_name is None:
        raise ValueError(
            f"Architecture or Encoder not found in checkpoint. Arch: {arch_name}, Encoder: {encoder_name}"
        )

    print(f"Architecture: {arch_name}")
    print(f"Encoder: {encoder_name}")

    if arch_name == "unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )
    elif arch_name == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )
    elif arch_name == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )
    else:
        raise ValueError(f"Unsupported architecture: {arch_name}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, encoder_name


def predict_image(model, image, transform, device, use_tta=False):
    """
    Run inference on a single image (supports TTA).
    """
    augmented = transform(image=image)
    img_tensor = augmented["image"].to(device)

    if use_tta:
        imgs = [torch.rot90(img_tensor, k, (1, 2)) for k in range(4)]
        batch = torch.stack(imgs)

        with torch.no_grad():
            logits = model(batch)
            probs = torch.sigmoid(logits)

            probs_unrotated = [torch.rot90(probs[k], 4 - k, (1, 2)) for k in range(4)]

            mean_prob = torch.stack(probs_unrotated).mean(dim=0)
            pred = mean_prob.squeeze().cpu().numpy()

    else:
        img_tensor = img_tensor.unsqueeze(0)
        with torch.no_grad():
            logits = model(img_tensor)
            pred = torch.sigmoid(logits).squeeze().cpu().numpy()

    return pred


def parse_arguments(args=None):
    """Set up and parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Segmentation Model")

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("inputs/images/dataset"),
        help="Root dataset directory containing images/ and masks/ subdirs. "
        "Default: inputs/images/dataset.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to the model checkpoint to evaluate. Defaults to most recent in models/.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save evaluation logs. Default: models.",
    )

    # options
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable Test Time Augmentation (4x rotations). "
        "Slower. Testing both with / without gives indication if beneficial prediction time.",
    )

    return parser.parse_args(args)


def main(args):
    """Evaluate model on test set."""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    test_img_dir = args.dataset_dir / "images/test"
    test_mask_dir = args.dataset_dir / "masks/test"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.output_dir / "evaluate_results.log"

    if not test_img_dir.exists():
        print(f"Error: Test image directory not found: {test_img_dir}")
        return
    if not test_mask_dir.exists():
        print(f"Error: Test mask directory not found: {test_mask_dir}")
        return

    if args.model is None:
        models_dir = Path("models")
        if not models_dir.exists():
            print("Error: 'models/' directory not found. Cannot auto-detect model.")
            return

        files = list(models_dir.glob("*.pth"))
        if not files:
            print(f"Error: No .pth files found in {models_dir}")
            return

        # the checkpoint only used if no regular inference file (should not be the case)
        inference_files = [f for f in files if "_checkpoint.pth" not in f.name]
        candidates = inference_files if inference_files else files
        candidates.sort(key=lambda f: f.name)
        args.model = candidates[-1]
        print(f"No model specified. Using most recent: {args.model}")

    if not args.model.exists():
        print(f"Error: Model checkpoint not found: {args.model}")
        return

    print(f"Loading model from {args.model}...")
    model, encoder_name = load_model(args.model, DEVICE)

    print(f"Loading test images from {test_img_dir}...")
    test_images = [
        f.name
        for f in test_img_dir.iterdir()
        if f.suffix == ".tif" and (test_mask_dir / f.name).exists()
    ]

    if len(test_images) == 0:
        print("Error: No test images found!")
        return

    print(f"Found {len(test_images)} test images.")

    transform = get_preprocessing()

    iou_scores = []
    dice_scores = []

    print(f"Evaluating model on test set (TTA={args.tta})...")
    for img_name in tqdm(test_images, desc="Processing test images"):
        # load image
        img_path = test_img_dir / img_name
        # load mask
        mask_path = test_mask_dir / img_name

        with suppress_stderr():
            image = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), 0)

        if image is None:
            print(f"\n[CRITICAL ERROR] Could not read IMAGE at: {img_path}")
            print(
                f"File size: {img_path.stat().st_size if img_path.exists() else 'Missing'} bytes"
            )
            raise ValueError(f"Corrupt file found: {img_path}")

        if mask is None:
            print(f"\n[CRITICAL ERROR] Could not read MASK at: {mask_path}")
            print(
                f"File size: {mask_path.stat().st_size if mask_path.exists() else 'Missing'} bytes"
            )
            raise ValueError(f"Corrupt file found: {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (mask > 0).astype(np.float32)

        pred = predict_image(model, image, transform, DEVICE, use_tta=args.tta)

        iou = calculate_iou(pred, mask)
        dice = calculate_dice(pred, mask)

        iou_scores.append(iou)
        dice_scores.append(dice)

    # convert to numpy arrays
    iou_scores = np.array(iou_scores)
    dice_scores = np.array(dice_scores)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_name = args.model.name
    tta_status = "Enabled" if args.tta else "Disabled"

    results_text = []
    results_text.append("=" * 60)
    results_text.append(f"EVALUATION RESULTS - {timestamp}")
    results_text.append("=" * 60)
    results_text.append(f"Model: {model_name}")
    results_text.append(f"Encoder: {encoder_name}")
    results_text.append(f"TTA (Test Time Augmentation): {tta_status}")
    results_text.append(f"Total test images: {len(test_images)}")
    results_text.append("")
    results_text.append("IoU Scores:")
    results_text.append(f"  Mean:   {iou_scores.mean():.4f}")
    results_text.append(f"  Median: {np.median(iou_scores):.4f}")
    results_text.append(f"  Min:    {iou_scores.min():.4f}")
    results_text.append(f"  Max:    {iou_scores.max():.4f}")
    results_text.append(f"  Std:    {iou_scores.std():.4f}")
    results_text.append("")
    results_text.append("Dice Scores:")
    results_text.append(f"  Mean:   {dice_scores.mean():.4f}")
    results_text.append(f"  Median: {np.median(dice_scores):.4f}")
    results_text.append(f"  Min:    {dice_scores.min():.4f}")
    results_text.append(f"  Max:    {dice_scores.max():.4f}")
    results_text.append(f"  Std:    {dice_scores.std():.4f}")
    results_text.append("=" * 60)
    results_text.append("")

    print("\n" + "\n".join(results_text))

    # append to log file
    with open(log_file, "a") as f:
        f.write("\n".join(results_text) + "\n")

    print(f"Results appended to {log_file}")
    print("=" * 60)

    return iou_scores, dice_scores


if __name__ == "__main__":
    parsed_args = parse_arguments()
    main(parsed_args)
