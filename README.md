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