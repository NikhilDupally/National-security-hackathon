import os
import numpy as np
import rasterio
from rasterio.transform import from_origin

# Ensure the images directory exists
os.makedirs('data/images', exist_ok=True)

# Define an example geotransform (origin at top-left coordinates, pixel size 0.002 degrees)
transform = from_origin(-122.5, 37.8, 0.002, 0.002)

# Create a "before" image: all zeros
before = np.zeros((100, 100), dtype=np.uint8)

# Create an "after" image: introduce a rectangular change patch
after = before.copy()
after[30:60, 40:70] = 50  # change intensity in the patch

# Write the "before" GeoTIFF
with rasterio.open(
    'data/images/test-area_before.tif',
    'w',
    driver='GTiff',
    height=100,
    width=100,
    count=1,
    dtype='uint8',
    crs='EPSG:4326',
    transform=transform
) as dst:
    dst.write(before, 1)

# Write the "after" GeoTIFF
with rasterio.open(
    'data/images/test-area_after.tif',
    'w',
    driver='GTiff',
    height=100,
    width=100,
    count=1,
    dtype='uint8',
    crs='EPSG:4326',
    transform=transform
) as dst:
    dst.write(after, 1)

print("Sample GeoTIFFs created at:")
print(" - /mnt/data/images/test-area_before.tif")
print(" - /mnt/data/images/test-area_after.tif")
