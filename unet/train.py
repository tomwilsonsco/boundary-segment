"""Train a segmentation model (U-Net / UNet++ / DeepLabV3+ / FPN) on chip images and binary boundary masks."""

import logging
import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import cv2
import segmentation_models_pytorch as smp
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from torch.amp import autocast, GradScaler
from contextlib import contextmanager
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@contextmanager
def suppress_stderr():
    """
    Low-level context manager to suppress C-level stderr output (like libtiff warnings).
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


class FieldDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.ids = [
            f.name
            for f in self.img_dir.iterdir()
            if f.suffix == ".tif" and (self.mask_dir / f.name).exists()
        ]

    def __getitem__(self, i):
        img_name = self.ids[i]
        img_path = self.img_dir / img_name
        mask_path = self.mask_dir / img_name

        with suppress_stderr():
            image = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), 0)

        if image is None:
            logging.error(f"\n[CRITICAL ERROR] Could not read IMAGE at: {img_path}")
            logging.error(
                f"File size: {img_path.stat().st_size if img_path.exists() else 'Missing'} bytes"
            )
            raise ValueError(f"Corrupt file found: {img_path}")

        if mask is None:
            logging.error(f"Could not read MASK at: {mask_path}")
            raise ValueError(f"Corrupt file found: {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask[mask > 0] = 1.0

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        return image, mask.float()

    def __len__(self):
        return len(self.ids)


def get_training_augmentation():
    """Return the albumentations pipeline used for training: flips, rotation, affine, brightness, and ImageNet normalisation."""
    return albu.Compose(
        [
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomRotate90(p=0.5),
            albu.Affine(
                scale=(0.9, 1.1), rotate=(-15, 15), translate_percent=(-0.1, 0.1), p=0.5
            ),
            albu.RandomBrightnessContrast(p=0.2),
            # imageNet normalization
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_validation_augmentation():
    """Return the albumentations pipeline used for validation: ImageNet normalisation only (no geometric augmentation)."""
    return albu.Compose(
        [
            # imageNet normalization
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def parse_arguments(args=None):
    """Set up and parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Segmentation Model")

    # data
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("inputs/images/dataset"),
        help="Root dataset directory containing images/ and masks/ subdirs. "
        "Default: inputs/images/dataset.",
    )

    parser.add_argument(
        "--loss-method",
        type=str,
        default="bce_dice",
        choices=["focal_dice", "bce_dice", "bce_only"],
        help="Loss function combination to use. Default: bce_dice.",
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=10.0,
        help="Positive class weight for BCE loss (bce_dice and bce_only). "
        "Set to approx. background_pixels / boundary_pixels ratio. Default: 10.0.",
    )

    # model
    parser.add_argument(
        "--arch",
        type=str,
        default="unetplusplus",
        choices=["unet", "unetplusplus", "deeplabv3plus", "fpn"],
        help="Model architecture. Default: unetplusplus.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="efficientnet-b3",
        help="Encoder backbone (e.g. efficientnet-b3, resnet34). "
        "See the Segmentation Models PyTorch documentation for all available encoders. Default: efficientnet-b3.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="imagenet",
        help="Pre-trained weight source for the encoder backbone. Changing this is not recommended. "
        "ImageNet weights give the best starting point for aerial imagery. Default: imagenet.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs to train for. Default: 30.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for training. Default: 8."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of workers for DataLoader. Defaults to min(os.cpu_count(), 4).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Starting learning rate for the AdamW optimiser. Default: 0.0001.",
    )
    parser.add_argument(
        "--accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps. Not used if 1. Default: 1.",
    )

    parser.add_argument(
        "--lr-patience",
        type=int,
        default=3,
        help="Patience for learning rate scheduler (in epochs). "
        "Will halve LR if no improvement in val loss after this many epochs. "
        "Default: 3.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help="Patience for early stopping (in epochs). "
        "Will stop training if no improvement in val loss after this many epochs. "
        "Default: 10.",
    )
    parser.add_argument(
        "--desc",
        type=str,
        default="rgb_025m",
        help="Experiment description. Included in names of output model pth files. "
        "Default: rgb_025m.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save trained model files and the training loss plot. Default: models.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional path to checkpoint to resume from.",
    )
    parser.add_argument(
        "--no-compile",
        dest="compile",
        action="store_false",
        help="Disable torch.compile. Enabled by default for faster training on CUDA.",
    )
    parser.set_defaults(compile=True)
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help="Use BF16 mixed precision instead of FP16. Recommended for Ada/Ampere+ GPUs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. Sets Python, NumPy, and PyTorch seeds. "
        "Also disables cuDNN benchmarking to ensure deterministic op selection. "
        "Omit to use a non-deterministic run (faster on GPU).",
    )

    return parser.parse_args(args)


def _seed_worker(worker_id):
    """Ensure each DataLoader worker gets a unique but deterministic seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Seeding — must happen before model init, DataLoader creation, etc.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logging.info(
            f"Seed set to {args.seed} (deterministic mode enabled, cuDNN benchmarking disabled)"
        )

    if torch.cuda.is_available():
        if args.seed is None:
            torch.backends.cudnn.benchmark = True
            logging.info("cuDNN benchmarking enabled")
        torch.set_float32_matmul_precision("high")
        logging.info("TF32 matmul precision enabled")

    # ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # generate timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arch_name = args.arch.lower()

    # include experiment description in filenames
    if args.desc:
        base_name = f"{timestamp}_{args.desc}_{arch_name}"
    else:
        base_name = f"{timestamp}_{arch_name}"

    model_save_path = args.output_dir / f"{base_name}.pth"
    checkpoint_path = args.output_dir / f"{base_name}_checkpoint.pth"
    loss_plot_path = args.output_dir / f"{base_name}_loss.png"

    logging.info(f"Experiment: {args.desc if args.desc else 'N/A'}")
    logging.info(f"Model architecture: {args.arch}")
    logging.info(f"Model will be saved to: {model_save_path}")
    logging.info(f"Checkpoint will be saved to: {checkpoint_path}")
    logging.info(f"Loss plot will be saved to: {loss_plot_path}")

    train_dataset = FieldDataset(
        args.dataset_dir / "images/train",
        args.dataset_dir / "masks/train",
        transform=get_training_augmentation(),
    )
    val_dataset = FieldDataset(
        args.dataset_dir / "images/val",
        args.dataset_dir / "masks/val",
        transform=get_validation_augmentation(),
    )

    # determine number of workers for DataLoader
    num_cores = os.cpu_count() or 1
    if args.num_workers is None:
        num_workers = min(num_cores, 4)
    else:
        num_workers = args.num_workers
        if num_workers > num_cores:
            logging.warning(
                f"Warning: --num-workers ({num_workers}) is greater than available cores ({num_cores}). "
                f"Using {num_cores} to avoid resource overallocation."
            )
            num_workers = num_cores

    logging.info(f"Using {num_workers} workers for data loading.")

    generator = torch.Generator()
    if args.seed is not None:
        generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=_seed_worker if args.seed is not None else None,
        generator=generator if args.seed is not None else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=_seed_worker if args.seed is not None else None,
    )

    logging.info(
        f"Training on {len(train_dataset)} images, Validating on {len(val_dataset)} images."
    )

    # Select model architecture
    model_params = {
        "encoder_name": args.encoder,
        "encoder_weights": args.weights,
        "in_channels": 3,
        "classes": 1,
        "activation": None,
    }

    if args.arch == "unet":
        model = smp.Unet(**model_params)
    elif args.arch == "unetplusplus":
        model = smp.UnetPlusPlus(**model_params)
    elif args.arch == "deeplabv3plus":
        model = smp.DeepLabV3Plus(**model_params)
    elif args.arch == "fpn":
        model = smp.FPN(**model_params)
    else:
        raise ValueError(
            f"Unknown architecture: {args.arch}. Supported: unet, unetplusplus, deeplabv3plus, fpn"
        )

    logging.info(f"Using {args.arch} architecture with {args.encoder} encoder")
    model.to(DEVICE)

    if args.compile and DEVICE == "cuda":
        logging.info("Compiling model with torch.compile (reduce-overhead mode)...")
        model = torch.compile(model, mode="reduce-overhead")
        logging.info("Model compiled.")

    if args.loss_method == "bce_only":
        logging.info(f"Using loss method: BCE only (pos_weight={args.pos_weight})")
        bce_only_loss = smp.losses.SoftBCEWithLogitsLoss(
            pos_weight=torch.tensor([args.pos_weight]).to(DEVICE)
        )
        dice_loss = None
        secondary_loss = None
    else:
        dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
        if args.loss_method == "bce_dice":
            logging.info(
                f"Using loss method: BCE + Dice (pos_weight={args.pos_weight})"
            )
            secondary_loss = smp.losses.SoftBCEWithLogitsLoss(
                pos_weight=torch.tensor([args.pos_weight]).to(DEVICE)
            )
        else:
            logging.info("Using loss method: Focal + Dice")
            secondary_loss = smp.losses.FocalLoss(mode="binary", alpha=0.85, gamma=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.lr_patience,
        threshold=0.001,
        threshold_mode="rel",
    )
    logging.info(f"Initial learning rate: {args.lr}")
    logging.info(
        f"LR scheduler: ReduceLROnPlateau (factor=0.5, patience={args.lr_patience}, threshold=0.1% rel improvement)"
    )

    # Initialize training state
    start_epoch = 0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    # Check if we should resume from existing checkpoint
    if args.resume and args.resume.exists():
        logging.info(f"\nResuming from checkpoint: {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location=DEVICE)

            # Check if it's a full checkpoint or just weights
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint["epoch"] + 1
                best_val_loss = checkpoint["best_val_loss"]
                epochs_no_improve = checkpoint["epochs_no_improve"]
                train_losses = checkpoint["train_losses"]
                val_losses = checkpoint["val_losses"]
                logging.info(
                    f"Successfully loaded checkpoint from epoch {checkpoint['epoch']}"
                )
                logging.info(
                    f"Resuming from epoch {start_epoch} with best val loss: {best_val_loss:.4f}"
                )
                logging.info(f"Will save to new checkpoint: {checkpoint_path}\n")
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
                logging.info("Loaded model weights only (from inference model file).")
                logging.info(
                    "Optimizer, learning rate, and epochs will start from scratch.\n"
                )
            else:
                model.load_state_dict(checkpoint)
                logging.info(
                    "Loaded model weights only (old format - no optimizer state)"
                )
                logging.warning("Optimizer will restart from scratch\n")
        except Exception as e:
            logging.warning(f"Could not load checkpoint: {e}")
            logging.info("Starting training from scratch with pretrained encoder...\n")
    elif args.resume:
        logging.warning(f"\nCheckpoint specified but not found: {args.resume}")
        logging.info("Starting training from scratch with pretrained encoder...\n")
    else:
        logging.info("\nStarting training from scratch with pretrained encoder...\n")

    # use_amp only if using gpu
    use_amp = DEVICE == "cuda"
    if use_amp and args.bf16:
        amp_dtype = torch.bfloat16
        scaler = GradScaler(enabled=False)
        logging.info("Mixed precision training enabled (BF16)")
    elif use_amp:
        amp_dtype = torch.float16
        scaler = GradScaler(enabled=True)
        logging.info("Mixed precision training enabled (FP16)")
    else:
        amp_dtype = torch.float32
        scaler = GradScaler(enabled=False)
        logging.info("Training with FP32 (CPU mode)")

    if args.accum_steps > 1:
        logging.info(
            f"Gradient accumulation enabled: {args.accum_steps} steps (effective batch size: {args.batch_size * args.accum_steps})"
        )

    logging.info(f"Early stopping patience: {args.early_stop_patience} epochs")
    for epoch in range(start_epoch, args.epochs):

        model.train()
        train_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")

        for batch_idx, (images, masks) in enumerate(loop):
            images = images.to(
                DEVICE, non_blocking=True
            )  # non_blocking with pin_memory
            masks = masks.to(DEVICE, non_blocking=True)

            with autocast(DEVICE, dtype=amp_dtype, enabled=use_amp):
                logits = model(images)
                if args.loss_method == "bce_only":
                    loss = bce_only_loss(logits, masks)
                else:
                    loss = dice_loss(logits, masks) + secondary_loss(logits, masks)
                loss = loss / args.accum_steps

            scaler.scale(loss).backward()

            if ((batch_idx + 1) % args.accum_steps == 0) or (
                batch_idx + 1 == len(train_loader)
            ):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * args.accum_steps  # unscale for logging
            loop.set_postfix(loss=loss.item() * args.accum_steps)

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0

        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        with torch.no_grad():
            for images, masks in val_loop:
                images = images.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)

                # mixed precision validation
                with autocast(DEVICE, dtype=amp_dtype, enabled=use_amp):
                    logits = model(images)
                    if args.loss_method == "bce_only":
                        loss = bce_only_loss(logits, masks)
                    else:
                        loss = dice_loss(logits, masks) + secondary_loss(logits, masks)

                val_loss += loss.item()
                val_loop.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # step learning rate scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        logging.info(
            f"Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}"
        )

        if avg_val_loss < best_val_loss:
            logging.info(
                f"Validation Loss improved ({best_val_loss:.4f} -> {avg_val_loss:.4f}). Saving checkpoint..."
            )
            best_val_loss = avg_val_loss
            epochs_no_improve = 0

            # save full checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "epochs_no_improve": epochs_no_improve,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "arch": args.arch,
                "encoder": args.encoder,
                "loss_method": args.loss_method,
                "batch_size": args.batch_size,
            }
            torch.save(checkpoint, checkpoint_path)
            # save model weights and config for inference
            inference_config = {
                "state_dict": model.state_dict(),
                "arch": args.arch,
                "encoder": args.encoder,
            }
            torch.save(inference_config, model_save_path)
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement for {epochs_no_improve} epoch(s)")

            if epochs_no_improve >= args.early_stop_patience:
                logging.info(f"\nEarly stopping triggered after {epoch+1} epochs")
                logging.info(f"Best validation loss: {best_val_loss:.4f}")
                break

        logging.info("-" * 30)

    logging.info(f"Training Complete. Best Validation Loss: {best_val_loss:.4f}")

    # plot / save training curves
    logging.info(f"Saving training loss plot to {loss_plot_path}...")
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses, "b-", label="Training Loss", linewidth=2)
    plt.plot(epochs_range, val_losses, "r-", label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training and Validation Loss", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Training loss plot saved successfully.")


if __name__ == "__main__":
    parsed_args = parse_arguments()
    main(parsed_args)
