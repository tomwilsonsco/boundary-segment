# run_test_pipeline.sh
# run the full pipeline covered in the README.md on a dataset for testing purposes.
# uses minimal training parameters (unet, 1 epoch, efficientnet-b0) for speed.

set -e # exit is non 0 status

# adjust to relative paths
SOURCE_IMAGES_DIR="inputs/images/gretna/12.5cm Aerial Photo"
PARCELS_GPKG="inputs/gretna_parcels.gpkg"

# output locations
EXP_NAME="test_quick_run"
OUTPUT_ROOT="outputs/${EXP_NAME}"

# intermediate directories 
TIFF_DIR="${SOURCE_IMAGES_DIR}/tiff_with_crs"
DOWNSCALE_DIR="${TIFF_DIR}/downscaled_025"
CHIPS_DIR="${DOWNSCALE_DIR}/chips"

# training / prediction directories
DATASET_DIR="${OUTPUT_ROOT}/dataset"
MODEL_DIR="${OUTPUT_ROOT}/models"

echo "======================================================="
echo "Starting Full Pipeline Test: ${EXP_NAME}"
echo "======================================================="

# 1. Assign CRS and convert to Tiff
echo "[Step 1] Assigning CRS and converting JPEGs to TIFF..."
python utils/assign_crs_to_images.py \
    --img-dir "${SOURCE_IMAGES_DIR}" \
    --output-subdir "tiff_with_crs" \
    --target-crs "EPSG:27700"

# 2. Downscale
echo "[Step 2] Downscaling images to 0.25m..."
python utils/downscale.py \
    --img-dir "${TIFF_DIR}" \
    --output-subdir "downscaled_025"

# 3. Create VRT
echo "[Step 3] Creating VRT mosaic..."
python utils/create_vrt.py \
    --img-dir "${DOWNSCALE_DIR}"

# Detect the VRT file created (Assuming one VRT is created in the dir)
VRT_FILE=$(ls "${DOWNSCALE_DIR}"/*.vrt | head -n1)
if [ -z "$VRT_FILE" ]; then
    echo "Error: No VRT file found in ${DOWNSCALE_DIR}"
    exit 1
fi
echo "VRT created: ${VRT_FILE}"

# 4. Chip Image
echo "[Step 4] Chipping VRT into tiles..."
python utils/chip_image.py \
    --vrt "${VRT_FILE}" \
    --output-subdir "chips" \
    --chip-size 512 \
    --overwrite-output-dir

# 5. Create Masks
# Creates binary masks in ${CHIPS_DIR}/masks
echo "[Step 5] Creating segmentation masks..."
python unet/create_masks.py \
    --chip-dir "${CHIPS_DIR}" \
    --shapefile "${PARCELS_GPKG}" \
    --buffer-size 0.75

# 6. Split Dataset
echo "[Step 6] Splitting dataset..."
python unet/split_dataset_train_test.py \
    --image-dir "${CHIPS_DIR}" \
    --mask-dir "${CHIPS_DIR}/masks" \
    --output-dir "${OUTPUT_ROOT}" \
    --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2

# 7. Train Model
# Using efficientnet-b0 and 1 epoch for speed
echo "[Step 7] Training model..."
python unet/train.py \
    --dataset-dir "${DATASET_DIR}" \
    --arch unet \
    --encoder efficientnet-b0 \
    --epochs 1 \
    --batch-size 4 \
    --output-dir "${MODEL_DIR}" \
    --desc "${EXP_NAME}"

# # Detect the trained model path (ignoring the checkpoint file)
MODEL_PATH=$(ls -t "${MODEL_DIR}"/*_${EXP_NAME}_*.pth | grep -v "checkpoint" | head -n1)
echo "Using trained model: ${MODEL_PATH}"

# 8. Evaluate
echo "[Step 8] Evaluating model..."
python unet/evaluate.py \
    --dataset-dir "${DATASET_DIR}" \
    --model "${MODEL_PATH}" \
    --output-dir "${OUTPUT_ROOT}/eval"

# 9. Predict
# Predicting on the chips folder generated in Step 4
echo "[Step 9] Running prediction..."
python unet/predict.py \
    --input-dir "${CHIPS_DIR}" \
    --model "${MODEL_PATH}" \
    --output-dir "${OUTPUT_ROOT}/predictions"

# 10. Example Plots
echo "[Step 10] Generating analysis plots..."
python unet/example_plots.py \
    --dataset-dir "${DATASET_DIR}" \
    --parcels-gpkg "${PARCELS_GPKG}" \
    --model "${MODEL_PATH}" \
    --output-dir "${OUTPUT_ROOT}/plots" \
    --num-samples 5

echo "Pipeline test complete. Outputs in ${OUTPUT_ROOT}"
