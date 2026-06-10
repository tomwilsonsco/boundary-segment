# run_test_pipeline.sh
# run the full pipeline covered in the README.md on a dataset for testing purposes.

set -euo pipefail

# adjust to relative paths
SOURCE_IMAGES_DIR="inputs/images/gretna/12.5cm Aerial Photo"
PARCELS_GPKG="inputs/gretna_parcels.gpkg"

# output locations
EXP_NAME="test_quick_run"
OUTPUT_ROOT="outputs/${EXP_NAME}"

# Setup logging
mkdir -p "${OUTPUT_ROOT}"
LOG_FILE="${OUTPUT_ROOT}/pipeline.log"
exec > >(tee "${LOG_FILE}") 2>&1
echo "Logging output to ${LOG_FILE}"

# intermediate directories 
TIFF_DIR="${SOURCE_IMAGES_DIR}/tiff_with_crs"

# DOWNSCALE_DIR="${TIFF_DIR}/downscaled_025"
CHIPS_DIR="${TIFF_DIR}/chips"

# training / prediction directories
DATASET_DIR="${OUTPUT_ROOT}/dataset"
MODEL_DIR="${OUTPUT_ROOT}/models"

echo "======================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Full Pipeline Test: ${EXP_NAME}"
echo "======================================================="

# 1. Assign CRS and convert to Tiff
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Step 1] Assigning CRS and converting JPEGs to TIFF..."
python utils/assign_crs_to_images.py \
    --img-dir "${SOURCE_IMAGES_DIR}" \
    --output-subdir "tiff_with_crs" \
    --crs "EPSG:27700"


# 2. Create VRT
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Step 2] Creating VRT mosaic..."
python utils/create_vrt.py \
    --img-dir "${TIFF_DIR}"

# Detect the VRT file created (Assuming one VRT is created in the dir)
VRT_FILE=$(find "${TIFF_DIR}" -maxdepth 1 -name "*.vrt" | head -n1)
if [ -z "$VRT_FILE" ]; then
    echo "Error: No VRT file found in ${TIFF_DIR}"
    exit 1
fi
echo "VRT created: ${VRT_FILE}"

# 3. Chip Image
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Step 3] Chipping VRT into tiles..."
python utils/chip_image.py \
    --vrt "${VRT_FILE}" \
    --output-subdir "chips" \
    --chip-size 512 \
    --chip-offset 384 \
    --resampling-factor 0.5 \
    --overwrite-output-dir

# 4. Create Masks
# Creates binary masks in ${CHIPS_DIR}/masks
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Step 4] Creating segmentation masks..."
python unet/create_masks.py \
    --chip-dir "${CHIPS_DIR}" \
    --parcels "${PARCELS_GPKG}" \
    --buffer-dist 0.75

# 5. Split Dataset
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Step 5] Splitting dataset..."
python unet/split_dataset_train_test.py \
    --chip-dir "${CHIPS_DIR}" \
    --mask-dir "${CHIPS_DIR}/masks" \
    --output-dir "${OUTPUT_ROOT}" \
    --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1

# 6. Train Model
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Step 6] Training model..."
python unet/train.py \
    --dataset-dir "${DATASET_DIR}" \
    --arch unetplusplus \
    --encoder efficientnet-b0 \
    --epochs 2 \
    --batch-size 8 \
    --num-workers 8 \
    --output-dir "${MODEL_DIR}" \
    --desc "${EXP_NAME}" \
    --bf16

# Detect the trained model path (ignoring the checkpoint file)
MODEL_PATH=$(ls -t "${MODEL_DIR}"/*_${EXP_NAME}_*.pth | grep -v "checkpoint" | head -n1)
if [ -z "$MODEL_PATH" ]; then
    echo "Error: No trained model found in ${MODEL_DIR}"
    exit 1
fi
echo "Using trained model: ${MODEL_PATH}"

# 7. Predict
# Predicting on the chips folder generated in Step 3
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Step 7] Running prediction..."
python unet/predict.py \
    --chip-dir "${CHIPS_DIR}" \
    --model "${MODEL_PATH}" \
    --output-dir "${OUTPUT_ROOT}/predictions" \
    --num-workers 8

# Detect prediction GPKG
PRED_GPKG=$(ls -t "${OUTPUT_ROOT}/predictions"/*_boundaries*.gpkg | head -n1)
if [ -z "$PRED_GPKG" ]; then
    echo "Error: No prediction GPKG found in ${OUTPUT_ROOT}/predictions"
    exit 1
fi
echo "Using prediction GPKG: ${PRED_GPKG}"

# 8. Run line evaluation
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Step 8] Running line evaluation..."
python unet/line_evaluate.py \
    --pred-gpkg "${PRED_GPKG}" \
    --parcels "${PARCELS_GPKG}" \
    --chip-dir "${CHIPS_DIR}" \
    --buffer-dist 3

# Detect line comparison GPKG
COMPARE_GPKG="${PRED_GPKG%.gpkg}_result_compare.gpkg"

# 9. Stats per chip
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Step 9] Calculating chip metrics..."
python unet/chip_metrics.py \
    --line-comparison "${COMPARE_GPKG}" \
    --mask-dir "${CHIPS_DIR}/masks" \
    --chips-index "${CHIPS_DIR}/chips_index.gpkg" \
    --dataset-dir "${DATASET_DIR}" \
    --output-gpkg "${CHIPS_DIR}/chips_index_metrics.gpkg"

# 10. Filter training chips
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Step 10] Filtering training chips..."
python unet/filter_training_chips.py \
    --input-gpkg "${CHIPS_DIR}/chips_index_metrics.gpkg" \
    --chip-dir "${CHIPS_DIR}" \
    --min-training-length 30.0 \
    --recall-min 0.5 \
    --min-precision 0.5

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline test complete. Outputs in ${OUTPUT_ROOT}"