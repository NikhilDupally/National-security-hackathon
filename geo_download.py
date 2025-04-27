# geodownload_sf.py

import ee
import json
import geemap

# 1) Authenticate & initialize Earth Engine (run once interactively)
#    ee.Authenticate()  
ee.Initialize(project="abcde")

# 2) Load AOI from GeoJSON
with open("sf_aoi.json") as f:
    gj = json.load(f)
aoi = ee.Geometry(gj)

# 3) Define helper to fetch & export median‐composite
def export_year(year, max_cloud=20):
    start = f"{year}-01-01"
    end   = f"{year}-12-31"
    col = (ee.ImageCollection("COPERNICUS/S2")
           .filterDate(start, end)
           .filterBounds(aoi)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud)))
    img = col.median().clip(aoi)
    out_fname = f"sf_s2_{year}.tif"
    print(f"Exporting {year} → {out_fname}  (this may take a few minutes)…")
    geemap.ee_export_image(
        img,
        filename=out_fname,
        scale=10,           # 10 m Sentinel-2
        region=aoi,
        file_per_band=False
    )

# 4) Run for 2016 and 2017
export_year(2016)
export_year(2017)
