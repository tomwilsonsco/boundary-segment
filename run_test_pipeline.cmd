@echo off
setlocal EnableDelayedExpansion

REM run_test_pipeline.cmd
REM Windows equivalent of run_test_pipeline.sh for running the full ML pipeline.

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
set "DOWNSCALE_DIR=%TIFF_DIR%\downscaled_025"
set "CHIPS_DIR=%DOWNSCALE_DIR%\chips"

REM Training / Prediction directories
set "DATASET_DIR=%OUTPUT_ROOT%\dataset"
set "MODEL_DIR=%OUTPUT_ROOT%\models"

echo =======================================================
echo Starting Full Pipeline Test: %EXP_NAME%
echo =======================================================

REM 1. Assign CRS and convert to Tiff
echo [Step 1] Assigning CRS and converting JPEGs to TIFF...
python utils\assign_crs_to_images.py --img-dir "%SOURCE_IMAGES_DIR%" --output-subdir "tiff_with_crs" --target-crs "EPSG:27700"
if %errorlevel% neq 0 exit /b %errorlevel%

REM 2. Downscale
echo [Step 2] Downscaling images to 0.25m...
python utils\downscale.py --img-dir "%TIFF_DIR%" --output-subdir "downscaled_025"
if %errorlevel% neq 0 exit /b %errorlevel%

REM 3. Create VRT
echo [Step 3] Creating VRT mosaic...
python utils\create_vrt.py --img-dir "%DOWNSCALE_DIR%"
if %errorlevel% neq 0 exit /b %errorlevel%

REM Detect the VRT file created (find the first .vrt file)
set "VRT_FILE="
for %%f in ("%DOWNSCALE_DIR%\*.vrt") do set "VRT_FILE=%%f"

if not defined VRT_FILE (
    echo Error: No VRT file found in %DOWNSCALE_DIR%
    exit /b 1
)
echo VRT created: %VRT_FILE%

REM 4. Chip Image
echo [Step 4] Chipping VRT into tiles...
python utils\chip_image.py --vrt "%VRT_FILE%" --output-subdir "chips" --chip-size 512 --overwrite-output-dir
if %errorlevel% neq 0 exit /b %errorlevel%

REM 5. Create Masks
echo [Step 5] Creating segmentation masks...
python unet\create_masks.py --chip-dir "%CHIPS_DIR%" --shapefile "%PARCELS_GPKG%" --buffer-size 0.75
if %errorlevel% neq 0 exit /b %errorlevel%

REM 6. Split Dataset
echo [Step 6] Splitting dataset...
python unet\split_dataset_train_test.py --image-dir "%CHIPS_DIR%" --mask-dir "%CHIPS_DIR%\masks" --output-dir "%OUTPUT_ROOT%" --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2
if %errorlevel% neq 0 exit /b %errorlevel%

REM 7. Train Model
echo [Step 7] Training model...
python unet\train.py --dataset-dir "%DATASET_DIR%" --arch unet --encoder efficientnet-b0 --epochs 1 --batch-size 4 --output-dir "%MODEL_DIR%" --desc "%EXP_NAME%"
if %errorlevel% neq 0 exit /b %errorlevel%

REM Detect the trained model path
REM We look for files matching the pattern, sort by date (newest first), and ignore 'checkpoint' files.
set "MODEL_PATH="
for /f "delims=" %%f in ('dir "%MODEL_DIR%\*_%EXP_NAME%_*.pth" /b /o:-d') do (
    echo %%f | find /i "checkpoint" >nul
    if errorlevel 1 (
        set "MODEL_PATH=%MODEL_DIR%\%%f"
        goto :found_model
    )
)
:found_model

if not defined MODEL_PATH (
    echo Error: Could not determine trained model path in %MODEL_DIR%
    exit /b 1
)
echo Using trained model: %MODEL_PATH%

REM 8. Evaluate
echo [Step 8] Evaluating model...
python unet\evaluate.py --dataset-dir "%DATASET_DIR%" --model "%MODEL_PATH%" --output-dir "%OUTPUT_ROOT%\eval"
if %errorlevel% neq 0 exit /b %errorlevel%

REM 9. Predict
echo [Step 9] Running prediction...
python unet\predict.py --input-dir "%CHIPS_DIR%" --model "%MODEL_PATH%" --output-dir "%OUTPUT_ROOT%\predictions"
if %errorlevel% neq 0 exit /b %errorlevel%

REM 10. Example Plots
echo [Step 10] Generating analysis plots...
python unet\example_plots.py --dataset-dir "%DATASET_DIR%" --parcels-gpkg "%PARCELS_GPKG%" --model "%MODEL_PATH%" --output-dir "%OUTPUT_ROOT%\plots" --num-samples 5

echo Pipeline test complete. Outputs in %OUTPUT_ROOT%