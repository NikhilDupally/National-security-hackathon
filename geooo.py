import os
import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import openai
from tqdm import tqdm

# ─── Configuration ─────────────────────────────────────────────────────────────

# Paths to your data files
BEFORE_PATH = "before.tif"
AFTER_PATH  = "after.tif"
ROADS_PATH  = "roads.geojson"
INTEL_PATH  = "intel.txt"    # optional

# Threshold for detecting change (sum over bands)
CHANGE_THRESHOLD = 50

# OpenAI settings
openai.api_key = os.getenv("OPENAI_API_KEY")

# ─── 1. Load imagery and compute diff mask ──────────────────────────────────────

with rasterio.open(BEFORE_PATH) as src1, rasterio.open(AFTER_PATH) as src2:
    assert src1.shape == src2.shape, "Rasters must match shape"
    before = src1.read().astype(np.int16)
    after  = src2.read().astype(np.int16)

# pixel-wise absolute difference summed over bands
diff = np.abs(after - before).sum(axis=0)
mask = (diff > CHANGE_THRESHOLD).astype(np.uint8)

# ─── 2. Polygonize mask ────────────────────────────────────────────────────────

# shapes(...) yields (geometry, value) pairs
polys = []
for geom, val in shapes(mask, transform=src1.transform):
    if val == 1:
        polys.append(shape(geom))

gdf_changes = gpd.GeoDataFrame({"geometry": polys}, crs=src1.crs)

# ─── 3. Contextualize with historical roads ─────────────────────────────────────

roads = gpd.read_file(ROADS_PATH).to_crs(gdf_changes.crs)
# Tag any change polygon that intersects a road
gdf_changes["type"] = [
    "road_damage" if roads.intersects(poly).any() else "ground_change"
    for poly in gdf_changes.geometry
]

# ─── 4. (Optional) Load and trim intel ──────────────────────────────────────────

intel_text = ""
if os.path.exists(INTEL_PATH):
    with open(INTEL_PATH, "r", encoding="utf-8") as f:
        intel_text = f.read()

# ─── 5. Build prompt and call LLM ───────────────────────────────────────────────

# Summarize polygon counts
summary_lines = []
for t in gdf_changes["type"].unique():
    cnt = (gdf_changes["type"] == t).sum()
    summary_lines.append(f"- {cnt} regions of {t.replace('_',' ')}")

coords_list = [
    poly.centroid.coords[0] for poly in gdf_changes.geometry
]
coord_snippet = ", ".join(f"({x:.4f},{y:.4f})" for x,y in coords_list[:5])
if len(coords_list) > 5:
    coord_snippet += ", …"

prompt = (
    "You are an intelligence analyst. A geospatial AI system has detected changes:\n"
    f"{chr(10).join(summary_lines)}\n"
    f"Sample locations: {coord_snippet}\n\n"
    "Relevant intelligence text:\n"
    f"{intel_text[:500]}…\n\n"
    "Provide a concise (3–5 sentence) strategic report highlighting likely causes and suggested next steps."
)

resp = openai.ChatCompletion.create(
    model="gpt-4o-mini",
    messages=[{"role":"system","content":"You are concise and factual."},
              {"role":"user","content":prompt}],
    temperature=0.3,
)

report = resp.choices[0].message.content.strip()

# ─── 6. Save outputs ────────────────────────────────────────────────────────────

gdf_changes.to_file("detected_changes.geojson", driver="GeoJSON")
with open("report.txt","w") as f:
    f.write(report)

print("► GeoJSON written to detected_changes.geojson")
print("► Report written to report.txt\n")
print("=== LLM Report ===")
print(report)
