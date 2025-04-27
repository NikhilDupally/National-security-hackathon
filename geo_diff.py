import os
from dotenv import load_dotenv

import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape

from anthropic import Anthropic

# ─── CONFIG & AUTH ──────────────────────────────────────────────────────────────
load_dotenv()  # loads ANTHROPIC_API_KEY from .env

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise RuntimeError(
        "Missing ANTHROPIC_API_KEY in environment. "
        "Add it to a .env file or export it."
    )

client = Anthropic(api_key=api_key)

BEFORE_PATH      = "data/20160110.tif"
AFTER_PATH       = "data/20170104.tif"
CHANGE_THRESHOLD = 50
MODEL            = "claude-3-opus-20240229"


# ─── 1. LOAD RASTERS ─────────────────────────────────────────────────────────────
with rasterio.open(BEFORE_PATH) as src1:
    before    = src1.read().astype(np.int16)
    transform = src1.transform
    crs       = src1.crs

with rasterio.open(AFTER_PATH) as src2:
    after = src2.read().astype(np.int16)
    # assume same shape & CRS


# ─── 2. COMPUTE DIFF & MASK ──────────────────────────────────────────────────────
diff = np.abs(after - before).sum(axis=0)          # (H, W)
mask = (diff > CHANGE_THRESHOLD).astype(np.uint8)  # binary mask


# ─── 3. POLYGONIZE CHANGES ──────────────────────────────────────────────────────
polys = []
for geom, val in shapes(mask, mask=mask, transform=transform):
    if val == 1:
        polys.append(shape(geom))

gdf = gpd.GeoDataFrame({"geometry": polys}, crs=crs)


# ─── 4. CALL ANTHROPIC MESSAGES API ───────────────────────────────────────────────
count   = len(gdf)
coords  = [poly.centroid.coords[0] for poly in gdf.geometry]
snippet = ", ".join(f"({x:.4f},{y:.4f})" for x,y in coords[:5])
if len(coords) > 5:
    snippet += ", …"

messages = [
    # {
    #     "role": "system",
    #     "content": "You are a concise and factual geospatial analyst."
    # },
    {
        "role": "user",
        "content":
            "An automated pipeline compared two geospatial images and detected:\n"
            f"- {count} change regions between the two dates.\n"
            f"Sample locations: {snippet}\n\n"
            "Please write a concise (3–5 sentence) strategic summary, "
            "suggesting likely causes and next steps."
    },
]

resp = client.messages.create(
    model=MODEL,
    messages=messages,
    max_tokens=300,
    # temperature=0.0,
)

report = resp.content[0].text.strip()


# ─── 5. SAVE OUTPUTS ─────────────────────────────────────────────────────────────
geojson_out = "detected_changes.geojson"
report_out  = "change_report.txt"

gdf.to_file(geojson_out, driver="GeoJSON")
with open(report_out, "w") as f:
    f.write(report)

print(f"► Wrote polygons → {geojson_out}")
print(f"► Wrote summary → {report_out}\n")
print("=== Anthropic Report ===")
print(report)
