@echo off
setlocal EnableDelayedExpansion

REM run_test_pipeline.cmd
REM Windows equivalent of run_test_pipeline.sh for running the full ML pipeline.
REM Runs the 10 core steps described in README.md.

REM --- Configuration ---
REM Adjust these paths to point to your actual test inputs
REM Note: Use quotes if paths contain spaces
set "SOURCE_IMAGES_DIR=inputs\images\gretna\12.5cm Aerial Photo"
set "PARCELS_GPKG=inputs\gretna_parcels.gpkg"

REM Output locations
set "EXP_NAME=test_quick_run"
set "OUTPUT_ROOT=outputs\%EXP_NAME%"

REM Intermediate directories
set "TIFF_DIR=%SOURCE_IMAGES_DIR%\tiff_with_crs"
set "CHIPS_DIR=%TIFF_DIR%\chips"

REM Training / Prediction directories
set "DATASET_DIR=%OUTPUT_ROOT%\dataset"
set "MODEL_DIR=%OUTPUT_ROOT%\models"

REM Setup logging
if not exist "%OUTPUT_ROOT%" mkdir "%OUTPUT_ROOT%"
set "LOG_FILE=%OUTPUT_ROOT%\pipeline.log"

echo ======================================================= >> "%LOG_FILE%"
echo Starting Full Pipeline Test: %EXP_NAME% >> "%LOG_FILE%"
echo ======================================================= >> "%LOG_FILE%"
echo =======================================================
echo Starting Full Pipeline Test: %EXP_NAME%
echo =======================================================

REM 1. Assign CRS and convert to Tiff
echo [%DATE% %TIME%] [Step 1] Assigning CRS and converting JPEGs to TIFF... >> "%LOG_FILE%"
echo [Step 1] Assigning CRS and converting JPEGs to TIFF...
python utils\assign_crs_to_images.py --img-dir "%SOURCE_IMAGES_DIR%" --output-subdir "tiff_with_crs" --crs "EPSG:27700"
if %errorlevel% neq 0 exit /b %errorlevel%

REM 2. Create VRT
echo [%DATE% %TIME%] [Step 2] Creating VRT mosaic... >> "%LOG_FILE%"
echo [Step 2] Creating VRT mosaic...
python utils\create_vrt.py --img-dir "%TIFF_DIR%"
if %errorlevel% neq 0 exit /b %errorlevel%

REM Detect the VRT file created (find the first .vrt file)
set "VRT_FILE="
for %%f in ("%TIFF_DIR%\*.vrt") do (
    if not defined VRT_FILE set "VRT_FILE=%%f"
)

if not defined VRT_FILE (
    echo Error: No VRT file found in %TIFF_DIR%
    exit /b 1
)
echo VRT created: %VRT_FILE% >> "%LOG_FILE%"
echo VRT created: %VRT_FILE%

REM 3. Chip Image
echo [%DATE% %TIME%] [Step 3] Chipping VRT into tiles... >> "%LOG_FILE%"
echo [Step 3] Chipping VRT into tiles...
python utils\chip_image.py --vrt "%VRT_FILE%" --output-subdir "chips" --chip-size 512 --chip-offset 384 --resampling-factor 0.5 --overwrite-output-dir --sample-scaler
if %errorlevel% neq 0 exit /b %errorlevel%

REM 4. Create Masks
echo [%DATE% %TIME%] [Step 4] Creating segmentation masks... >> "%LOG_FILE%"
echo [Step 4] Creating segmentation masks...
python unet\create_masks.py --chip-dir "%CHIPS_DIR%" --parcels "%PARCELS_GPKG%" --buffer-dist 0.75
if %errorlevel% neq 0 exit /b %errorlevel%

REM 5. Split Dataset
echo [%DATE% %TIME%] [Step 5] Splitting dataset... >> "%LOG_FILE%"
echo [Step 5] Splitting dataset...
python unet\split_dataset_train_test.py --chip-dir "%CHIPS_DIR%" --mask-dir "%CHIPS_DIR%\masks" --output-dir "%OUTPUT_ROOT%" --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1
if %errorlevel% neq 0 exit /b %errorlevel%

REM 6. Train Model
echo [%DATE% %TIME%] [Step 6] Training model... >> "%LOG_FILE%"
echo [Step 6] Training model...
python unet\train.py --dataset-dir "%DATASET_DIR%" --arch unetplusplus --encoder efficientnet-b0 --epochs 1 --batch-size 8 --num-workers 8 --output-dir "%MODEL_DIR%" --desc "%EXP_NAME%" --bf16
if %errorlevel% neq 0 exit /b %errorlevel%

REM Detect the trained model path (ignoring the checkpoint file)
set "MODEL_PATH="
for /f "delims=" %%f in ('dir "%MODEL_DIR%\*_%EXP_NAME%_*.pth" /b /o:-d') do (
    REM Check if the file name contains "checkpoint"
    echo %%f | find /i "checkpoint" >nul
    if errorlevel 1 (
        set "MODEL_PATH=%MODEL_DIR%\%%f"
        goto :found_model
    )
)
:found_model

if not defined MODEL_PATH (
    echo Error: No trained model found in %MODEL_DIR%
    exit /b 1
)
echo Using trained model: %MODEL_PATH% >> "%LOG_FILE%"
echo Using trained model: %MODEL_PATH%

REM 7. Predict
REM Predicting on the chips folder generated in Step 3
echo [%DATE% %TIME%] [Step 7] Running prediction... >> "%LOG_FILE%"
echo [Step 7] Running prediction...
python unet\predict.py --chip-dir "%CHIPS_DIR%" --model "%MODEL_PATH%" --output-dir "%OUTPUT_ROOT%\predictions" --num-workers 8
if %errorlevel% neq 0 exit /b %errorlevel%

REM Detect prediction GPKG
set "PRED_GPKG="
for /f "delims=" %%f in ('dir "%OUTPUT_ROOT%\predictions\*_boundaries*.gpkg" /b /o:-d') do (
    set "PRED_GPKG=%OUTPUT_ROOT%\predictions\%%f"
    goto :found_pred_gpkg
)
:found_pred_gpkg

if not defined PRED_GPKG (
    echo Error: No prediction GPKG found in %OUTPUT_ROOT%\predictions
    exit /b 1
)
echo Using prediction GPKG: !PRED_GPKG! >> "%LOG_FILE%"
echo Using prediction GPKG: !PRED_GPKG!

REM 8. Run line evaluation
echo [%DATE% %TIME%] [Step 8] Running line evaluation... >> "%LOG_FILE%"
echo [Step 8] Running line evaluation...
python unet\line_evaluate.py --pred-gpkg "!PRED_GPKG!" --parcels "%PARCELS_GPKG%" --chip-dir "%CHIPS_DIR%" --buffer-dist 3
if %errorlevel% neq 0 exit /b %errorlevel%

REM Build comparison GPKG path by replacing .gpkg with _result_compare.gpkg
set "COMPARE_GPKG=!PRED_GPKG:.gpkg=_result_compare.gpkg!"

REM 9. Stats per chip
echo [%DATE% %TIME%] [Step 9] Calculating chip metrics... >> "%LOG_FILE%"
echo [Step 9] Calculating chip metrics...
python unet\chip_metrics.py --line-comparison "!COMPARE_GPKG!" --mask-dir "%CHIPS_DIR%\masks" --chips-index "%CHIPS_DIR%\chips_index.gpkg" --dataset-dir "%DATASET_DIR%" --output-gpkg "%CHIPS_DIR%\chips_index_metrics.gpkg"
if %errorlevel% neq 0 exit /b %errorlevel%

REM 10. Filter training chips
echo [%DATE% %TIME%] [Step 10] Filtering training chips... >> "%LOG_FILE%"
echo [Step 10] Filtering training chips...
python unet\filter_training_chips.py --input-gpkg "%CHIPS_DIR%\chips_index_metrics.gpkg" --chip-dir "%CHIPS_DIR%" --min-training-length 30.0 --recall-min 0.5 --min-precision 0.5
if %errorlevel% neq 0 exit /b %errorlevel%

echo [%DATE% %TIME%] Pipeline test complete. Outputs in %OUTPUT_ROOT% >> "%LOG_FILE%"
echo Pipeline test complete. Outputs in %OUTPUT_ROOT%

endlocal