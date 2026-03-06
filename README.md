# Boundary Segment

1. ## Assign CRS and convert jpegs to geotiffs:
```bash
python utils/assign_crs_to_images.py --img-dir "inputs/images/gretna/12.5cm Aerial Photo"
```

2. ## Downscale 0.125m per pixel images to 0.25m
```bash
python utils/downscale.py --img-dir "inputs/images/gretna/12.5cm Aerial Photo/tiff_with_crs" --output-subdir downscaled_025
```

3. ## Create a GDAL vrt from the images
```bash
python utils/create_vrt.py --img-dir "inputs/images/gretna/12.5cm Aerial Photo/tiff_with_crs/downscaled_025"
```

4. ## Create chip images from vrt
Uses the [rs-chip](https://github.com/tomwilsonsco/rs-chip) package.  
Progress bar takes a while to move from 0 as only moves once whole batch complete. Look at the output dir that chip files are being created if unsure.
```bash
python utils/chip_image.py --vrt "inputs/images/gretna/12.5cm Aerial Photo/tiff_with_crs/
downscaled_025/apgb_imgs.vrt"
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
python unet/train.py --dataset-dir inputs/images/gretna/dataset --arch unetplusplus --encoder efficientnet-b3 --epochs 30 --batch-size 8 --lr 0.0001 --desc test-025m
```