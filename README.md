# Boundary Segment

1. ## Assign CRS and convert jpegs to geotiffs:
```bash
python utils/assign_crs_to_images.py --img-dir "inputs/images/gretna/12.5cm Aerial Photo"
```

2. ## Downscale 0.125m per pixel images to 0.25m
```bash
python utils/downscale.py --img-dir "inputs/images/gretna/12.5cm Aerial Photo/tiff_with_crs" --output-subdir downscaled_025
```