"""
run_pipeline.py

A wrapper script to run the full Land Parcel Boundary Segmentation pipeline.
Reads configuration from a YAML file and executes the underlying tools.

Each run writes its outputs to outputs/predictions/<run_id>/, where run_id is
<timestamp>_<area>. That folder is kept by the cleanup step and holds what is
needed to compare runs or to rebuild the area's training set later: both
predictions, their evaluations, a copy of the config, the chip ignore list and
chip metrics, the training loss plot and a summary.json.
"""

import argparse
import csv
import hashlib
import json
import logging
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# run every script with the interpreter running this wrapper, so conda / venv
# environments do not pick up a different Python
PYTHON = sys.executable


def load_config(config_path: str) -> dict:
    """Load the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Configuration file not found: {path}")
        sys.exit(1)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_command(cmd_list: list, step_name: str) -> None:
    """Run a command via subprocess and stop the pipeline if it fails.

    Args:
        cmd_list:    Command and arguments to run.
        step_name:   Human-readable label used in log messages.
    """
    cmd_str_list = [str(arg) for arg in cmd_list]
    logger.info(f"--- Running {step_name} ---")
    logger.debug(f"Command: {' '.join(cmd_str_list)}")

    try:
        subprocess.run(cmd_str_list, check=True, text=True)
        logger.info(f"--- {step_name} completed successfully ---\n")
    except subprocess.CalledProcessError as e:
        logger.error(f"Step '{step_name}' failed with exit code {e.returncode}.")
        sys.exit(1)


def require_output(path: Path, step_name: str) -> None:
    """Stop the pipeline if a step did not write the file the next step needs."""
    if not path.exists():
        logger.error(f"Step '{step_name}' did not produce expected output: {path}")
        sys.exit(1)


def file_record(path: Path) -> dict:
    """Path, size and SHA-256 of an input file, so a run records which version it used."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def ignore_reasons(ignore_csv: Path) -> dict:
    """Count chips in chips_ignore.csv by removal reason (a chip can have several)."""
    counts = Counter()
    with open(ignore_csv, newline="") as f:
        for row in csv.DictReader(f):
            for reason in (row.get("remove_condition") or "").split(" | "):
                counts[reason.strip() or "unspecified"] += 1
    return dict(counts)


def evaluate_command(
    pred_gpkg,
    parcels_gpkg,
    chips_dir,
    buffer_dist,
    output_gpkg,
    metrics_json,
    prediction_mask,
):
    """Build a line_evaluate.py command with explicit output paths."""
    cmd = [
        PYTHON,
        "unet/line_evaluate.py",
        "--pred-gpkg",
        pred_gpkg,
        "--parcels",
        parcels_gpkg,
        "--chip-dir",
        chips_dir,
        "--buffer-dist",
        buffer_dist,
        "--output-gpkg",
        output_gpkg,
        "--metrics-json",
        metrics_json,
    ]
    if prediction_mask:
        cmd.extend(["--prediction-mask", prediction_mask])
    return cmd


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
    started = datetime.now()

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

    base_model_path = Path(config["models"]["base_model"])

    # Buffers: the filter buffer scores chips for removal from training; the
    # report buffer is the business rule used for the metrics in summary.json.
    # A config with the old single 'buffer_dist' key uses it as the filter buffer.
    p_line_eval = config["parameters"].get("line_evaluate", {})
    filter_buffer = p_line_eval.get(
        "filter_buffer_dist", p_line_eval.get("buffer_dist", 3.0)
    )
    report_buffer = p_line_eval.get("report_buffer_dist", 8.0)

    # Every file passed between steps is named here, up front, so no step has to
    # guess which file the previous one wrote. All of a run's outputs except the
    # model go in one run folder.
    run_id = f"{started.strftime('%Y%m%d_%H%M%S')}_{area_name}"
    run_dir = Path("outputs/predictions") / run_id
    fine_tuned_desc = config["models"]["fine_tuned_desc"]
    arch = config["parameters"]["train"]["arch"]
    base_pred_gpkg = run_dir / f"{run_id}_base_boundaries.gpkg"
    filter_eval_gpkg = run_dir / f"{run_id}_base_filter_result_compare.gpkg"
    base_eval_gpkg = run_dir / f"{run_id}_base_result_compare.gpkg"
    fine_tuned_model_path = Path("models") / f"{run_id}_{fine_tuned_desc}_{arch}.pth"
    final_pred_gpkg = run_dir / f"{run_id}_{fine_tuned_desc}_boundaries.gpkg"
    final_eval_gpkg = run_dir / f"{run_id}_{fine_tuned_desc}_result_compare.gpkg"
    filter_metrics_json = run_dir / "metrics_base_filter.json"
    base_metrics_json = run_dir / "metrics_base.json"
    final_metrics_json = run_dir / "metrics_fine_tuned.json"

    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy(args.config, run_dir / "pipeline_config.yaml")
    logger.info(f"Starting pipeline for area: {area_name}")
    logger.info(f"Run folder: {run_dir}")

    # ==========================================
    # Stage 1: Data Preparation (Steps 1–4)

    # Step 1: Assign CRS and convert JPEGs to GeoTIFFs
    assign_crs_params = config["parameters"].get("assign_crs", {})
    cmd_step1 = [PYTHON, "utils/assign_crs_to_images.py", "--img-dir", raw_img_dir]
    if "crs" in assign_crs_params:
        cmd_step1.extend(["--crs", assign_crs_params["crs"]])
    if prediction_mask:
        cmd_step1.extend(["--prediction-mask", prediction_mask])
    run_command(cmd_step1, "Step 1: Assign CRS")

    # Step 2: Create GDAL VRT mosaic
    cmd_step2 = [PYTHON, "utils/create_vrt.py", "--img-dir", tiff_crs_dir]
    run_command(cmd_step2, "Step 2: Create VRT")

    # Step 3: Chip image
    # --overwrite-output-dir is required so the pipeline can be re-run without
    # manual cleanup; without it chip_image.py exits immediately if chips/ exists.
    p_chip = config["parameters"]["chip_image"]
    cmd_step3 = [
        PYTHON,
        "utils/chip_image.py",
        "--vrt",
        vrt_path,
        "--chip-size",
        p_chip["chip_size"],
        "--chip-offset",
        p_chip["chip_offset"],
        "--resampling-factor",
        p_chip["resampling_factor"],
        "--overwrite-output-dir",
    ]
    if prediction_mask:
        cmd_step3.extend(["--prediction-mask", prediction_mask])
    run_command(cmd_step3, "Step 3: Chip Image")

    # Step 4: Create binary mask TIFFs from land parcel polygons
    cmd_step4 = [
        PYTHON,
        "unet/create_masks.py",
        "--chip-dir",
        chips_dir,
        "--parcels",
        parcels_gpkg,
    ]
    run_command(cmd_step4, "Step 4: Create Masks")

    # ==========================================
    # Stage 2: Preliminary Dataset Split

    # chip_metrics.py (Step 9) requires background_only_check.csv, which is
    # produced by split_dataset_train_test.py / rschip.DatasetSplitter.  Run
    # an initial split now so that file exists.
    # The split is re-run after Step 10
    # with the updated chips_ignore.csv to produce the clean training dataset.

    cmd_step5_prelim = [
        PYTHON,
        "unet/split_dataset_train_test.py",
        "--chip-dir",
        chips_dir,
        "--mask-dir",
        masks_dir,
        "--output-dir",
        dataset_output_dir,
    ]
    run_command(cmd_step5_prelim, "Step 5 (Preliminary): Initial Dataset Split")

    # ==========================================
    # Stage 3: Initial Prediction & Filtering (Steps 7–10)

    # Step 7 (Initial): Predict with the pre-trained base model.
    cmd_step7_init = [
        PYTHON,
        "unet/predict.py",
        "--chip-dir",
        chips_dir,
        "--model",
        base_model_path,
        "--num-workers",
        8,
        "--output-dir",
        run_dir,
        "--output-gpkg",
        base_pred_gpkg,
    ]
    if prediction_mask:
        cmd_step7_init.extend(["--prediction-mask", prediction_mask])
    run_command(cmd_step7_init, "Step 7 (Initial): Predict with Base Model")
    require_output(base_pred_gpkg, "Step 7 (Initial): Predict with Base Model")

    # Step 8: Line Evaluate at the filter buffer, for chip filtering
    step = f"Step 8: Line Evaluate ({filter_buffer} m, for chip filtering)"
    run_command(
        evaluate_command(
            base_pred_gpkg,
            parcels_gpkg,
            chips_dir,
            filter_buffer,
            filter_eval_gpkg,
            filter_metrics_json,
            prediction_mask,
        ),
        step,
    )
    require_output(filter_eval_gpkg, step)

    # Step 8b: the base model has never seen this area, so this is its
    # out-of-sample score at the business-rule buffer
    step = f"Step 8b: Line Evaluate ({report_buffer} m, base model, out-of-sample)"
    run_command(
        evaluate_command(
            base_pred_gpkg,
            parcels_gpkg,
            chips_dir,
            report_buffer,
            base_eval_gpkg,
            base_metrics_json,
            prediction_mask,
        ),
        step,
    )
    require_output(base_metrics_json, step)

    # Step 9: Chip Metrics
    # --line-comparison: the TP/FP/FN comparison GPKG from line_evaluate
    # --mask-dir: directory containing background_only_check.csv
    # --output-gpkg: explicit path so Step 10 can locate it reliably
    cmd_step9 = [
        PYTHON,
        "unet/chip_metrics.py",
        "--line-comparison",
        filter_eval_gpkg,
        "--chips-index",
        chips_index_gpkg,
        "--mask-dir",
        masks_dir,
        "--output-gpkg",
        chips_metrics_gpkg,
        "--dataset-dir",
        dataset_dir,
    ]
    run_command(cmd_step9, "Step 9: Calculate Chip Metrics")

    # Step 10: Filter Training Chips — updates chips_ignore.csv
    p_filter = config["parameters"]["filter_chips"]
    cmd_step10 = [
        PYTHON,
        "unet/filter_training_chips.py",
        "--input-gpkg",
        chips_metrics_gpkg,
        "--chip-dir",
        chips_dir,
        "--min-training-length",
        p_filter["min_training_length"],
        "--recall-min",
        p_filter["recall_min"],
        "--min-precision",
        p_filter["min_precision"],
    ]
    run_command(cmd_step10, "Step 10: Filter Training Chips")

    # keep the ignore list and chip metrics with the run: together with the
    # config they define this area's training set, and cleanup deletes chips/
    ignore_csv = chips_dir / "chips_ignore.csv"
    require_output(ignore_csv, "Step 10: Filter Training Chips")
    shutil.copy(ignore_csv, run_dir / "chips_ignore.csv")
    shutil.copy(chips_metrics_gpkg, run_dir / "chips_index_metrics.gpkg")

    # ==========================================
    # Stage 4: Fine Prediction

    # Remove the preliminary dataset split so the final split can recreate it
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    # Step 5 (Final): Re-run dataset split now that chips_ignore.csv has been
    # updated by Step 10 — this produces the clean training dataset.
    cmd_step5_final = [
        PYTHON,
        "unet/split_dataset_train_test.py",
        "--chip-dir",
        chips_dir,
        "--mask-dir",
        masks_dir,
        "--output-dir",
        dataset_output_dir,
    ]
    run_command(cmd_step5_final, "Step 5 (Final): Create Cleaned Split Dataset")
    dataset_counts = {
        split: len(list((dataset_dir / "images" / split).glob("*.tif")))
        for split in ("train", "val", "test")
    }

    # Step 6: Train fine-tuned model.
    # --resume loads the base model weights as the starting point, implementing
    # the fine-tuning described in the README.
    p_train = config["parameters"]["train"]
    cmd_step6 = [
        PYTHON,
        "unet/train.py",
        "--dataset-dir",
        dataset_dir,
        "--arch",
        p_train["arch"],
        "--encoder",
        p_train["encoder"],
        "--loss-method",
        p_train["loss_method"],
        "--epochs",
        p_train["epochs"],
        "--batch-size",
        p_train["batch_size"],
        "--lr",
        p_train["lr"],
        "--model-path",
        fine_tuned_model_path,
        "--resume",
        base_model_path,
    ]
    if "early_stop_patience" in p_train:
        cmd_step6.extend(["--early-stop-patience", p_train["early_stop_patience"]])
    run_command(cmd_step6, "Step 6: Train Fine-Tuned Model")
    require_output(fine_tuned_model_path, "Step 6: Train Fine-Tuned Model")
    logger.info(f"Using fine-tuned model for final prediction: {fine_tuned_model_path}")
    loss_plot = fine_tuned_model_path.with_name(
        f"{fine_tuned_model_path.stem}_loss.png"
    )
    if loss_plot.exists():
        shutil.copy(loss_plot, run_dir / loss_plot.name)

    # Step 7 (Final): Predict with the fine-tuned model.
    cmd_step7_final = [
        PYTHON,
        "unet/predict.py",
        "--chip-dir",
        chips_dir,
        "--model",
        fine_tuned_model_path,
        "--output-dir",
        run_dir,
        "--output-gpkg",
        final_pred_gpkg,
    ]
    if prediction_mask:
        cmd_step7_final.extend(["--prediction-mask", prediction_mask])
    run_command(cmd_step7_final, "Step 7 (Final): Predict with Fine-Tuned Model")
    require_output(final_pred_gpkg, "Step 7 (Final): Predict with Fine-Tuned Model")
    logger.info(f"Final predictions: {final_pred_gpkg}")

    # Step 11: the fine-tuned model was trained on these labels, so this score
    # is in-sample; compare runs on the base model's out-of-sample score
    step = f"Step 11: Line Evaluate ({report_buffer} m, fine-tuned model, in-sample)"
    run_command(
        evaluate_command(
            final_pred_gpkg,
            parcels_gpkg,
            chips_dir,
            report_buffer,
            final_eval_gpkg,
            final_metrics_json,
            prediction_mask,
        ),
        step,
    )
    require_output(final_metrics_json, step)

    # ==========================================
    # Stage 5: Run summary

    summary = {
        "run_id": run_id,
        "area": area_name,
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "config": str(args.config),
        "inputs": {
            "parcels": file_record(parcels_gpkg),
            "prediction_mask": (
                file_record(prediction_mask) if prediction_mask else None
            ),
            "raw_image_dir": str(raw_img_dir),
        },
        "models": {
            "base": str(base_model_path),
            "fine_tuned": str(fine_tuned_model_path),
        },
        "chips": {
            "ignored_by_reason": ignore_reasons(run_dir / "chips_ignore.csv"),
            "dataset": dataset_counts,
        },
        "metrics": {
            "base_out_of_sample": json.loads(base_metrics_json.read_text()),
            "fine_tuned_in_sample": json.loads(final_metrics_json.read_text()),
            "base_at_filter_buffer": json.loads(filter_metrics_json.read_text()),
        },
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    base_m = summary["metrics"]["base_out_of_sample"]
    fine_m = summary["metrics"]["fine_tuned_in_sample"]
    logger.info(
        f"At {report_buffer} m: base (out-of-sample) P {base_m['precision']:.3f} "
        f"R {base_m['recall']:.3f} F1 {base_m['f1']:.3f}; fine-tuned (in-sample) "
        f"P {fine_m['precision']:.3f} R {fine_m['recall']:.3f} F1 {fine_m['f1']:.3f}"
    )
    logger.info(f"Run summary: {summary_path}")

    # ==========================================
    # Stage 6: Cleanup

    if not config["pipeline_settings"].get("preserve_intermediate_files", True):
        logger.info("Cleaning up intermediate files...")

        # Remove the GeoTIFFs, VRT, chips, and masks tree, plus the dataset split.
        # The original JPEG source files under raw_img_dir are intentionally kept,
        # and the run folder keeps what is needed to rebuild the training set.
        paths_to_remove = [
            tiff_crs_dir,  # contains tiffs, VRT, chips/, and masks/
            dataset_dir,  # contains the train/val/test split
        ]

        for p in paths_to_remove:
            if p.exists() and p.is_dir():
                logger.info(f"Removing directory: {p}")
                shutil.rmtree(p)

        logger.info(f"Cleanup complete. Run outputs remain in {run_dir}")
    else:
        logger.info("Intermediate files preserved.")

    logger.info("Pipeline Execution Complete!")


if __name__ == "__main__":
    main()
