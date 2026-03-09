# Boundary Segment
Predict land parcel boundaries using high resolution aerial photography and segmentation models such as [U-Net](https://arxiv.org/abs/1505.04597) and [Unet++](https://arxiv.org/abs/1807.10165).

# Setup
This process uses open source geospatial packages and PyTorch. Use of a GPU is recommended for training the models and making predictions. Can optionally use the Docker setup used in development of this process, found in `/.devcontainer`. 

## Using Docker
1. First cd to the repo and build the Docker image:
```bash
docker build -t boundary-segment .
```

2. Run the container (mounting current directory to /app):
```bash
docker run --gpus all -it --rm -v $(pwd):/app boundary-segment

# Windows Powershell use this instead:
docker run --gpus all -it --rm -v "${PWD}":/app boundary-segment
```

If you use VS Code and Docker a devcontainer.json is included in under `/.devcontainer`. In this case you do not need to run the docker commands above, just open the repository in VS Code and under Command Palette use Devcontainer: Reopen in container.

## Without Docker
The `requirements.txt` can be used to install the packages needed, if not on Windows.

**For Windows users** as the process uses GDAL, you will probably find it easier to install the geospatial packages with Conda. 

### Instructions for Windows users (non-Docker)
1. With Conda available on your machine, create a conda environment and install the geospatial packages with a conda install command.

```bash
# create conda geo_env and install gdal etc
conda create -n geo_env python=3.12 pip gdal rasterio geopandas shapely -c conda-forge
# activate the new env
conda activate geo_env
```

2. With the `geo_env` environment active, pip install the remaining packages into the conda python home.
```bash
# install torch
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# install other packages, e.g. rschip not on conda-forge
python -m pip install rschip segmentation-models-pytorch albumentations opencv-python-headless pytest black flake8 tqdm scikit-learn scikit-image matplotlib
```

# Training a segmentation model

Currently the process is available as a series of Python scripts with input arguments specified in the terminal. For all the scripts below, you can get the argparse command help.
```bash
python <path to script.py> --help
```

1. ## Assign CRS and convert jpegs to geotiffs:
The APGB images do not have a CRS and arrive in jpeg format.
```bash
python utils/assign_crs_to_images.py --img-dir "inputs/images/gretna/12.5cm Aerial Photo"
```

2. ## Downscale 0.125m per pixel images to 0.25m
For speed of process / chip tile size management it is recommended to downscale from 0.125 metre per pixel to 0.25.
```bash
python utils/downscale.py --img-dir "inputs/images/gretna/12.5cm Aerial Photo/tiff_with_crs" --output-subdir downscaled_025
```

3. ## Create a GDAL vrt from the images
A GDAL vrt is an XML reference file. It is much quicker to build and takes less disk space than building a mosaic image.
```bash
python utils/create_vrt.py --img-dir "inputs/images/gretna/12.5cm Aerial Photo/tiff_with_crs/downscaled_025"
```

4. ## Create chip images from vrt
Uses the [rs-chip](https://github.com/tomwilsonsco/rs-chip) package.  
Progress bar takes a while to move from 0 as only moves once whole batch complete. Look at the output dir that chip files are being created if unsure.
```bash
python utils/chip_image.py --vrt "inputs/images/gretna/12.5cm Aerial Photo/tiff_with_crs/downscaled_025/apgb_imgs.vrt"
```

5. ## Create masks from land parcel lines
This will create an equivalent binary mask tif (1 for lines 0 for background) for each input image.
```bash
python unet/create_masks.py --chip-dir "inputs/images/gretna/12.5cm Aerial Photo/tiff_with_crs/downscaled_025/chips" --shapefile inputs/gretna_parcels.gpkg
```
6. ## Split the chip images and masks into train, validation, test sets
The process using `rschip.DatasetSplitter()` will check for and not copy image-mask pairs that are all background (0 class only).

```bash
python unet/split_dataset_train_test.py --image-dir "inputs/images/gretna/12.5cm Aerial Photo/tiff_with_crs/downscaled_025/chips" --mask-dir "inputs/images/gretna/12.5cm Aerial Photo/tiff_with_crs/downscaled_025/chips/masks" --output-dir inputs/images/gretna
```

7. ## Train model
Many of these args are the default values but included here for info.
```bash
# to see help on script arguments
python unet/train.py --help

# running the training
python unet/train.py --dataset-dir inputs/images/gretna/dataset --arch unetplusplus --encoder efficientnet-b3 --epochs 30 --batch-size 8 --lr 0.0001 --desc test-025m
```

8. ## Evaluate model
We use the evaluate.py to use trained model to predict from each test set image and then report summary stats on intersect over union (IoU) and dice score.

```bash
python unet/evaluate.py --dataset-dir inputs/images/gretna/dataset --model models/20260309_094957_test-025m_unetplusplus.pth
```
 **Note:**: Although given above, we do not necessarily need to specify the model .pth file as by default it will take the most recent based on date time saved in the .pth file name. The `--model` argument is needed if not testing the most recently available.

9. ## Predict with model
Once a trained model is achieving test set prediction performance you are happy with, you can predict for all chipped images across a continuous extent and produce a geopackage output of the boundary line predictions.2

This process takes a while to complete on large extents.

```bash
python unet/predict.py --input-dir inputs/images/gretna/12.5cm Aerial Photo/tiff_with_crs/downscaled_025/chips
```

10. ## Plot some prediction examples
We can create plots as shown below for predictions on the test set of chips. Vary the number of samples and seed values to get different number and selection of plots.

```bash
python unet/example_plots.py --dataset-dir inputs/images/gretna/dataset --num-samples 5 --seed 999
```

![Example test set prediction](plots/apgb_imgs_8832_40320_analysis.png)