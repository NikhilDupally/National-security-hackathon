'''
  geospatial_change_detector.py

LLM-Powered Geospatial Change Detector integrating OSINT resources:
- AIS shipping data from AIS Hub and BarentsWatch
- Vessel traffic data dumps & Kaggle AIS dataset
- Flight tracking via Flightradar24
- Data visualization exports for deck.gl / kepler.gl
- FastAPI service for alerts and reports

Dependencies: requests, pandas, geopandas, shapely, rasterio, torch, faiss, sentence_transformers, openai, fastapi, uvicorn
'''
import os
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box, shape
import json
import rasterio
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# --- Configuration ---
AIS_HUB_API = "https://api.aishub.net/v1/ows"
AIS_HUB_KEY = os.getenv("AIS_HUB_API_KEY")
BARENTSWATCH_API = "https://api.barentswatch.no/v2/ais"
VTF_DUMP_PATH = "data/vessel_traffic.json"
KAGGLE_AIS_PATH = "data/kaggle_ais.csv"
FLIGHT_TRACKING_API = "https://api.flightradar24.com/common/v1/flight/list.json"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set your Area of Interest (AOI) here as GeoJSON
AOI_POLYGON = shape({
    "type": "Polygon",
    "coordinates": [
        [[-122.5,37.6],[-122.3,37.6],[-122.3,37.8],[-122.5,37.8],[-122.5,37.6]]
    ]
})

# --- Data Ingestors ---
class AISHubIngestor:
    def fetch(self, bbox):
        params = {
            "service": "AIS",
            "request": "GetAIS",
            "BBOX": ",".join(map(str, bbox)),
            "api_key": AIS_HUB_KEY
        }
        resp = requests.get(AIS_HUB_API, params=params)
        resp.raise_for_status()
        return pd.DataFrame(resp.json()["features"])

class BarentswatchIngestor:
    def fetch(self, bbox):
        lon_min, lat_min, lon_max, lat_max = bbox
        url = f"{BARENTSWATCH_API}?bbox={lon_min},{lat_min},{lon_max},{lat_max}"
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()["features"]
        rows = []
        for feat in data:
            rows.append({**feat["properties"]})
        return pd.DataFrame(rows)

class KaggleAISIngestor:
    def load(self):
        df = pd.read_csv(KAGGLE_AIS_PATH)
        # filter by AOI
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
        return gdf[gdf.geometry.within(AOI_POLYGON)]

class VesselTrafficIngestor:
    def load(self):
        with open(VTF_DUMP_PATH) as f:
            data = json.load(f)
        features = data.get("features", [])
        rows = [feat["properties"] for feat in features]
        df = pd.DataFrame(rows)
        return df

class FlightTracker:
    def fetch_recent(self):
        # Simple polling example: top flights in AOI
        params = {"lat":37.7, "lon":-122.4, "fDstL":50}
        resp = requests.get(FLIGHT_TRACKING_API, params=params)
        resp.raise_for_status()
        flights = resp.json()["data"]
        return pd.DataFrame(flights)

# --- Preprocessor ---
class Preprocessor:
    def clip_to_aoi(self, gdf):
        return gdf[gdf.geometry.within(AOI_POLYGON)]

# --- Change Detection (stub implementations) ---
class ChangeDetector:
    def __init__(self):
        # load your pre-trained PyTorch model here
        # self.model = torch.load(...) etc.
        pass
    def detect_changes(self, before_img_path, after_img_path):
        # stub: open rasters, compute difference mask
        with rasterio.open(before_img_path) as src1, rasterio.open(after_img_path) as src2:
            arr1 = src1.read(1)
            arr2 = src2.read(1)
        mask = (arr2.astype(float) - arr1.astype(float)) > 20  # thresholded change
        return mask.astype(int)

class KaggleAISIngestor:
    def __init__(self, path="data/ais_data_1.csv"):
        self.path = path

    def load(self):
        df = pd.read_csv(self.path)
        # Example: if lat/lon columns are named 'LATITUDE'/'LONGITUDE', rename them:
        df = df.rename(columns={"LAT":"lat", "LON":"lon"})
        gdf = gpd.GeoDataFrame(df,
                               geometry=gpd.points_from_xy(df.lon, df.lat),
                               crs="EPSG:4326")
        # Clip to your AOI polygon
        return gdf[gdf.geometry.within(AOI_POLYGON)]


# --- Feature Fusion ---
class FeatureFusion:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import faiss
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatL2(384)
        self.texts = []
    def add_text(self, text):
        emb = self.text_encoder.encode([text])
        self.index.add(emb)
        self.texts.append(text)
    def query(self, emb, top_k=3):
        D, I = self.index.search(emb, top_k)
        return [self.texts[i] for i in I[0]]

# --- LLM Summarizer ---
class LLMSummarizer:
    def __init__(self):
        import openai
        openai.api_key = OPENAI_API_KEY
    def summarize(self, location, change_desc, context_texts):
        prompt = (
            f"Location: {location}\n"
            f"Change: {change_desc}\n"
            f"Context: {'; '.join(context_texts)}\n"
            "Provide a concise intelligence brief:\n"
        )
        resp = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()

# --- FastAPI Service ---
app = FastAPI(title="Geospatial Change Detector API")

class WatchArea(BaseModel):
    id: str
    geojson: dict

@app.post("/ingest")
def ingest(area: WatchArea):
    # Compute the AOI bounding box
    bbox = gpd.GeoSeries([shape(area.geojson)]).total_bounds.tolist()
    # Load your Kaggle AIS dump and clip to AOI
    kag_df = KaggleAISIngestor().load()
    # (Optionally) also load a vessel-traffic JSON dump
    # vtf_df = VesselTrafficIngestor().load()
    # And fetch recent flights if you need them
    flights_df = FlightTracker().fetch_recent()

    # Persist to disk or PostGIS
    kag_df.to_file(f"data/ais_{area.id}.geojson", driver="GeoJSON")
    # vtf_df.to_csv(f"data/vtf_{area.id}.csv", index=False)
    flights_df.to_csv(f"data/flights_{area.id}.csv", index=False)

    return {"status": "ingested", "area": area.id}


@app.post("/detect/{area_id}")
def detect(area_id: str):
    # stub: use before/after satellite images from storage
    before = f"images/{area_id}_before.tif"
    after = f"images/{area_id}_after.tif"
    mask = ChangeDetector().detect_changes(before, after)
    out_png = f"outputs/change_mask_{area_id}.png"
    # save mask as PNG
    import matplotlib.pyplot as plt
    plt.imsave(out_png, mask)
    return {"mask_image": out_png}

@app.get("/report/{area_id}")
def report(area_id: str):
    # load masked changes and context
    location = "AOI coordinates"
    change_desc = "Detected significant pixel-level changes"
    fusion = FeatureFusion()
    # add dummy context
    fusion.add_text("NOTAM: new runway extension underway")
    texts = fusion.query(fusion.text_encoder.encode([change_desc]))
    summary = LLMSummarizer().summarize(location, change_desc, texts)
    return {"summary": summary, "context": texts}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
