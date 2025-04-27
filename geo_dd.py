import json
import os
import requests
from tqdm import tqdm

from pystac_client import Client
import planetary_computer
from shapely.geometry import shape, mapping
import rasterio

# ─── Config ─────────────────────────────────────────────────────────────────────

AOI_GEOJSON = "data/sf_aoi.json"
OUT_DIR     = "sf_s2_stac"
YEARS       = [2016, 2017]
MAX_CLOUD   = 20      # percent

# Make sure output dir exists
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 1. Load AOI ────────────────────────────────────────────────────────────────

with open(AOI_GEOJSON) as f:
    aoi = json.load(f)

# ─── 2. Connect to Planetary Computer STAC ──────────────────────────────────────

catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
collection_id = "sentinel-2-l2a"

# ─── 3. Search & Download ──────────────────────────────────────────────────────

for year in YEARS:
    time_range = f"{year}-01-01/{year}-12-31"
    print(f"\n🔍 Searching Sentinel-2 {year} (cloud<{MAX_CLOUD}%) …")
    
    search = catalog.search(
        collections=[collection_id],
        intersects=aoi,
        datetime=time_range,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}}
    )
    
    items = list(search.get_items())
    print(f"  → Found {len(items)} scenes")
    
    # For demonstration, we'll just grab up to 3 scenes per year
    for item in tqdm(items[:3], desc=f"Downloading {year}"):
        # sign URLs
        signed = planetary_computer.sign(item)
        # fetch the True-Color RGB (B04,B03,B02) composite asset
        # many STAC items have a “visual” asset; otherwise use B02/B03/B04 individually
        asset = signed.assets.get("visual") or signed.assets["B04"]
        
        href = asset.href
        fname = os.path.join(OUT_DIR, f"{year}_{item.id}_{os.path.basename(href)}")
        
        # skip if already downloaded
        if os.path.exists(fname):
            continue
        
        # stream download
        resp = requests.get(href, stream=True)
        resp.raise_for_status()
        with open(fname, "wb") as f:
            for chunk in resp.iter_content(1024*1024):
                f.write(chunk)
        
        # Optional: verify with rasterio
        with rasterio.open(fname) as src:
            print(f"   • {os.path.basename(fname)} → {src.width}×{src.height}, {src.count} bands")
    
print("\n✅ Done. Files in:", OUT_DIR)
