import argparse
import logging
import os
import shutil
import numpy as np
import rasterio
import torch
import geopandas as gpd
from shapely.geometry import Point
from rasterio.windows import Window
from tqdm import tqdm
from ultralytics import YOLO
from utils import download_blob, upload_blob, resample_raster

PATCH_SIZE = 256
STRIDE = 128
TARGET_RES = 0.10

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def predict_yolo_on_raster(input_raster_path, output_geojson_path, model_path, imgsz=256, conf=0.1):
    local_input = input_raster_path
    local_model = model_path
    
    if input_raster_path.startswith("gs://"):
        local_input = "/tmp/input_raster.tif"
        download_blob(input_raster_path, local_input)
        
    if model_path.startswith("gs://"):
        local_model = "/tmp/model.pt"
        download_blob(model_path, local_model)

    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO(local_model)

    all_detections = []

    with rasterio.open(local_input) as src:
        src_crs = src.crs
        src_nodata = src.nodata if src.nodata is not None else 32767.0
        
        for r_offset in tqdm(range(0, src.height, STRIDE)):
            for c_offset in range(0, src.width, STRIDE):
                window = Window(c_offset, r_offset, min(PATCH_SIZE, src.width-c_offset), min(PATCH_SIZE, src.height-r_offset))
                
                if window.width < PATCH_SIZE or window.height < PATCH_SIZE:
                    padded = np.full((PATCH_SIZE, PATCH_SIZE), src_nodata, dtype=np.float32)
                    d = src.read(1, window=window)
                    padded[:window.height, :window.width] = d
                    patch = padded
                else:
                    patch = src.read(1, window=window).astype(np.float32)

                # --- 8-bit Pre-processing (matches YOLO train.py) ---
                img_8bit = ((np.clip(patch, -300, 300) + 300) / 600 * 255).astype(np.uint8)
                img_8bit = np.stack([img_8bit, img_8bit, img_8bit], axis=-1) # YOLO expects 3 channels
                
                # YOLO expects RGB usually, even for grayscale we stack it or Ultralytics handles it
                results = model.predict(img_8bit, imgsz=imgsz, conf=conf, device=device, verbose=False)
                
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        # Box in pixel coords within patch
                        x_center, y_center = box.xywh[0][0].item(), box.xywh[0][1].item()
                        confidence = box.conf.item()
                        
                        # Convert to global pixel coords
                        global_x = c_offset + x_center
                        global_y = r_offset + y_center
                        
                        # Convert to CRS coords
                        lon, lat = src.xy(global_y, global_x)
                        all_detections.append({'geometry': Point(lon, lat), 'confidence': confidence})

    if not all_detections:
        logging.warning("No detections found.")
        gdf = gpd.GeoDataFrame(columns=['geometry', 'confidence'], crs=src_crs)
    else:
        gdf = gpd.GeoDataFrame(all_detections, crs=src_crs)
        
    gdf.to_file(output_geojson_path, driver='GeoJSON')
    logging.info(f"Saved {len(gdf)} detections to {output_geojson_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-raster", required=True)
    parser.add_argument("--output-geojson", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--conf", type=float, default=0.1)
    args = parser.parse_args()
    predict_yolo_on_raster(args.input_raster, args.output_geojson, args.model, args.imgsz, args.conf)
