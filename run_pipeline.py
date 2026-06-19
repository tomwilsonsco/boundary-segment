#!/usr/bin/env python3
"""
run_pipeline.py

A wrapper script to run the full Land Parcel Boundary Segmentation pipeline.
Reads configuration from a YAML file and executes the underlying tools.
"""

import argparse
import subprocess
import yaml
import sys
import shutil
import logging
from pathlib import Path

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Configuration file not found: {path}")
        sys.exit(1)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_command(cmd_list: list, step_name: str, stdin_input: str = None) -> None:
    """Run a command via subprocess and handle errors.

    Args:
        cmd_list:    Command and arguments to run.
        step_name:   Human-readable label used in log messages.
        stdin_input: Optional string written to the process stdin. Used to
                     automatically respond to interactive prompts (e.g. the
                     overwrite prompt in predict.py) so the pipeline never
                     blocks waiting for keyboard input.
    """
    cmd_str_list = [str(arg) for arg in cmd_list]
    logger.info(f"--- Running {step_name} ---")
    logger.debug(f"Command: {' '.join(cmd_str_list)}")

    try:
        subprocess.run(cmd_str_list, check=True, text=True, input=stdin_input)
        logger.info(f"--- {step_name} completed successfully ---\n")
    except subprocess.CalledProcessError as e:
        logger.error(f"Step '{step_name}' failed with exit code {e.returncode}.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run the full boundary segmentation pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pipeline_config.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # ------------------------------------------------------------------
    # Resolve all paths up-front so every step has consistent references
    # ------------------------------------------------------------------
    base_input = Path(config["data"]["base_input_dir"])
    area_name = config["data"]["area_name"]
    raw_img_dir = base_input / config["data"]["raw_image_dir"]
    parcels_gpkg = base_input / config["data"]["parcels_gpkg"]

    mask_val = config["data"].get("prediction_mask_gpkg")
    prediction_mask = base_input / mask_val if mask_val else None

    # Intermediate paths derived from the README directory structure
    tiff_crs_dir = raw_img_dir / "tiff_with_crs"
    vrt_path = tiff_crs_dir / "apgb_imgs.vrt"
    chips_dir = tiff_crs_dir / "chips"
    masks_dir = chips_dir / "masks"
    dataset_output_dir = base_input / "images" / area_name
    dataset_dir = dataset_output_dir / "dataset"
    chips_index_gpkg = chips_dir / "chips_index.gpkg"
    chips_metrics_gpkg = chips_dir / "chips_index_metrics.gpkg"

    base_model_name = config["models"]["base_model"]
    base_model_path = Path("models") / base_model_name
    predictions_dir = Path("outputs/predictions")

    logger.info(f"Starting pipeline for area: {area_name}")

    # ==========================================
    # Stage 1: Data Preparation (Steps 1–4)


    # Step 1: Assign CRS and convert JPEGs to GeoTIFFs
    assign_crs_params = config["parameters"].get("assign_crs", {})
    cmd_step1 = ["python", "utils/assign_crs_to_images.py", "--img-dir", raw_img_dir]
    if "crs" in assign_crs_params:
        cmd_step1.extend(["--crs", assign_crs_params["crs"]])
    if prediction_mask:
        cmd_step1.extend(["--prediction-mask", prediction_mask])
    run_command(cmd_step1, "Step 1: Assign CRS")

    # Step 2: Create GDAL VRT mosaic
    cmd_step2 = ["python", "utils/create_vrt.py", "--img-dir", tiff_crs_dir]
    run_command(cmd_step2, "Step 2: Create VRT")

    # Step 3: Chip image
    # --overwrite-output-dir is required so the pipeline can be re-run without
    # manual cleanup; without it chip_image.py exits immediately if chips/ exists.
    p_chip = config["parameters"]["chip_image"]
    cmd_step3 = [
        "python", "utils/chip_image.py",
        "--vrt", vrt_path,
        "--chip-size", p_chip["chip_size"],
        "--chip-offset", p_chip["chip_offset"],
        "--resampling-factor", p_chip["resampling_factor"],
        "--overwrite-output-dir",
    ]
    if prediction_mask:
        cmd_step3.extend(["--prediction-mask", prediction_mask])
    run_command(cmd_step3, "Step 3: Chip Image")

    # Step 4: Create binary mask TIFFs from land parcel polygons
    cmd_step4 = [
        "python", "unet/create_masks.py",
        "--chip-dir", chips_dir,
        "--parcels", parcels_gpkg,
    ]
    run_command(cmd_step4, "Step 4: Create Masks")

    # ==========================================
    # Stage 2: Preliminary Dataset Split
    #
    # chip_metrics.py (Step 9) requires background_only_check.csv, which is
    # produced by split_dataset_train_test.py / rschip.DatasetSplitter.  Run
    # an initial split now so that file exists.
    # The split is re-run after Step 10
    # with the updated chips_ignore.csv to produce the clean training dataset.


    cmd_step5_prelim = [
        "python", "unet/split_dataset_train_test.py",
        "--chip-dir", chips_dir,
        "--mask-dir", masks_dir,
        "--output-dir", dataset_output_dir,
    ]
    run_command(cmd_step5_prelim, "Step 5 (Preliminary): Initial Dataset Split")

    # ==========================================
    # Stage 3: Initial Prediction & Filtering (Steps 7–10)

    # Step 7 (Initial): Predict with the pre-trained base model.
    cmd_step7_init = [
        "python", "unet/predict.py",
        "--chip-dir", chips_dir,
        "--model", base_model_path,
    ]
    if prediction_mask:
        cmd_step7_init.extend(["--prediction-mask", prediction_mask])
    run_command(
        cmd_step7_init,
        "Step 7 (Initial): Predict with Base Model",
        stdin_input="o\n",
    )

    # Locate the prediction GPKG just written by predict.py (newest by time)
    pred_files = sorted(
        predictions_dir.glob("*.gpkg"), key=lambda f: f.stat().st_ctime
    )
    if not pred_files:
        logger.error(f"No prediction GPKG found in {predictions_dir}")
        sys.exit(1)
    latest_pred_gpkg = pred_files[-1]
    logger.info(f"Using prediction file for evaluation: {latest_pred_gpkg}")

    # Step 8: Line Evaluate — produces <pred_stem>_result_compare.gpkg
    p_line_eval = config["parameters"].get("line_evaluate", {})
    cmd_step8 = [
        "python", "unet/line_evaluate.py",
        "--pred-gpkg", latest_pred_gpkg,
        "--parcels", parcels_gpkg,
        "--chip-dir", chips_dir,
    ]
    if "buffer_dist" in p_line_eval:
        cmd_step8.extend(["--buffer-dist", p_line_eval["buffer_dist"]])
    if prediction_mask:
        cmd_step8.extend(["--prediction-mask", prediction_mask])
    run_command(cmd_step8, "Step 8: Line Evaluate")

    # line_evaluate.py always writes its output as <pred_stem>_result_compare.gpkg
    # in the same directory as the input prediction GPKG.
    eval_lines_gpkg = (
        latest_pred_gpkg.parent / f"{latest_pred_gpkg.stem}_result_compare.gpkg"
    )
    if not eval_lines_gpkg.exists():
        logger.error(
            f"Expected line evaluate output not found: {eval_lines_gpkg}"
        )
        sys.exit(1)

    # Step 9: Chip Metrics
    # --line-comparison: the TP/FP/FN comparison GPKG from line_evaluate
    # --mask-dir: directory containing background_only_check.csv 
    # --output-gpkg: explicit path so Step 10 can locate it reliably
    cmd_step9 = [
        "python", "unet/chip_metrics.py",
        "--line-comparison", eval_lines_gpkg,
        "--chips-index", chips_index_gpkg,
        "--mask-dir", masks_dir,
        "--output-gpkg", chips_metrics_gpkg,
        "--dataset-dir", dataset_dir,
    ]
    run_command(cmd_step9, "Step 9: Calculate Chip Metrics")

    # Step 10: Filter Training Chips — updates chips_ignore.csv
    p_filter = config["parameters"]["filter_chips"]
    cmd_step10 = [
        "python", "unet/filter_training_chips.py",
        "--input-gpkg", chips_metrics_gpkg,
        "--chip-dir", chips_dir,
        "--min-training-length", p_filter["min_training_length"],
        "--recall-min", p_filter["recall_min"],
        "--min-precision", p_filter["min_precision"],
    ]
    run_command(cmd_step10, "Step 10: Filter Training Chips")

    # ==========================================
    # Stage 4: Fine Prediction

    # Step 5 (Final): Re-run dataset split now that chips_ignore.csv has been
    # updated by Step 10 — this produces the clean training dataset.
    cmd_step5_final = [
        "python", "unet/split_dataset_train_test.py",
        "--chip-dir", chips_dir,
        "--mask-dir", masks_dir,
        "--output-dir", dataset_output_dir,
    ]
    run_command(cmd_step5_final, "Step 5 (Final): Create Cleaned Split Dataset")

    # Step 6: Train fine-tuned model.
    # --resume loads the base model weights as the starting point, implementing
    # the fine-tuning described in the README.
    p_train = config["parameters"]["train"]
    cmd_step6 = [
        "python", "unet/train.py",
        "--dataset-dir", dataset_dir,
        "--arch", p_train["arch"],
        "--encoder", p_train["encoder"],
        "--loss-method", p_train["loss_method"],
        "--epochs", p_train["epochs"],
        "--batch-size", p_train["batch_size"],
        "--lr", p_train["lr"],
        "--desc", config["models"]["fine_tuned_desc"],
        "--resume", base_model_path,
    ]
    run_command(cmd_step6, "Step 6: Train Fine-Tuned Model")

    # Locate the newly trained model: most-recently created .pth in models/,
    # excluding training checkpoints.  Using ctime rather than name sort avoids
    # the risk of the base model sorting alphabetically after the fine-tuned one.
    models_dir = Path("models")
    fine_tuned_candidates = sorted(
        [f for f in models_dir.glob("*.pth") if "_checkpoint.pth" not in f.name],
        key=lambda f: f.stat().st_ctime,
    )
    if not fine_tuned_candidates:
        logger.error("No fine-tuned model .pth found in models/ after training.")
        sys.exit(1)
    fine_tuned_model_path = fine_tuned_candidates[-1]
    logger.info(f"Using fine-tuned model for final prediction: {fine_tuned_model_path}")

    # Step 7 (Final): Predict with the fine-tuned model.
    cmd_step7_final = [
        "python", "unet/predict.py",
        "--chip-dir", chips_dir,
        "--model", fine_tuned_model_path,
    ]
    if prediction_mask:
        cmd_step7_final.extend(["--prediction-mask", prediction_mask])
    run_command(
        cmd_step7_final,
        "Step 7 (Final): Predict with Fine-Tuned Model",
        stdin_input="o\n",
    )

    # ==========================================
    # Stage 5: Cleanup

    if not config["pipeline_settings"].get("preserve_intermediate_files", True):
        logger.info("Cleaning up intermediate files...")

        # Remove the GeoTIFFs, VRT, chips, and masks tree, plus the dataset split.
        # The original JPEG source files under raw_img_dir are intentionally kept.
        paths_to_remove = [
            tiff_crs_dir,       # contains tiffs, VRT, chips/, and masks/
            dataset_dir,        # contains the train/val/test split
        ]

        for p in paths_to_remove:
            if p.exists() and p.is_dir():
                logger.info(f"Removing directory: {p}")
                shutil.rmtree(p)

        logger.info(
            "Cleanup complete. Final predictions remain in outputs/predictions/"
        )
    else:
        logger.info("Intermediate files preserved.")

    logger.info("Pipeline Execution Complete!")


if __name__ == "__main__":
    main()
