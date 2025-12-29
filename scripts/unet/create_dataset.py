import argparse
import logging
import os
import sys
import tempfile
import tarfile
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import io
import geopandas as gpd
import harmonica as hm
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter
from shapely.geometry import Point, box
from tqdm import tqdm
from google.cloud import storage

from utils import download_blob, upload_blob, resample_raster

# --- Configuration (v58: v38 improvements - Random Dipole & Smaller Radii) ---
PATCH_SIZE = 256
STRIDE = 128
NODATA_THRESHOLD = 0.5
FIELD_INCLINATION = 60.0
FIELD_DECLINATION = 0.0
GAUSSIAN_SIGMA = 64.0 

LOG_FORMAT = '%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt='%Y-%m-%d %H:%M:%S')

def shift_cibles_to_anomaly_peak(gdf_cibles, src_raster):
    pixel_size_m = src_raster.transform[0]
    search_radius_m = 10.0
    search_pixels = int(search_radius_m / pixel_size_m)
    rel_coords = np.linspace(-search_radius_m, search_radius_m, 2 * search_pixels + 1)
    rel_xx, rel_yy = np.meshgrid(rel_coords, rel_coords)
    rel_obs_points = np.array([rel_xx.ravel(), rel_yy.ravel(), np.zeros_like(rel_xx.ravel())])
    
    shifted_points = []
    
    for index, row in gdf_cibles.iterrows():
        center_e, center_n = row.geometry.x, row.geometry.y
        famille = row.get('famille', 'C')
        
        if famille == 'A':
            random_depth = random.uniform(-6.0, -2.5)
            random_magnitude = random.uniform(2000, 8000)
        elif famille == 'B':
            random_depth = random.uniform(-3.5, -1.0)
            random_magnitude = random.uniform(500, 2000)
        else:
            random_depth = random.uniform(-1.5, -0.2)
            random_magnitude = random.uniform(50, 500)
            
        dipole_pos = (center_e, center_n, random_depth)
        
        # v38 improvement: Randomize dipole orientation (remanent magnetization)
        mx = random.uniform(-1, 1)
        my = random.uniform(-1, 1)
        mz = random.uniform(-1, 1)
        norm = (mx**2 + my**2 + mz**2)**0.5 + 1e-9 
        DIPOLE_MOMENT = (
            (mx / norm) * random_magnitude,
            (my / norm) * random_magnitude,
            (mz / norm) * random_magnitude
        )
            
        obs_e = rel_obs_points[0] + center_e
        obs_n = rel_obs_points[1] + center_n
        obs_z = rel_obs_points[2]
        
        bx, by, bz = hm.dipole_magnetic((obs_e, obs_n, obs_z), dipole_pos, DIPOLE_MOMENT, field="b")
        tmi_anomaly = hm.total_field_anomaly((bx, by, bz), inclination=FIELD_INCLINATION, declination=FIELD_DECLINATION)
        
        peak_index = np.argmax(np.abs(tmi_anomaly))
        shifted_points.append(Point(obs_e[peak_index], obs_n[peak_index]))

    gdf_shifted = gdf_cibles.copy()
    gdf_shifted.geometry = shifted_points
    return gdf_shifted

def process_zone_and_archive(raster_blob_name, bucket_name, global_cibles_gdf, output_bucket, output_prefix):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        local_raster = tmp_path / Path(raster_blob_name).name
        download_blob(f"gs://{bucket_name}/{raster_blob_name}", str(local_raster))
        
        target_res = 0.10
        with rasterio.open(local_raster) as src:
            res_x, res_y = src.transform[0], abs(src.transform[4])
        
        if abs(res_x - target_res) > 1e-3:
            resampled_path = tmp_path / f"resampled_{local_raster.name}"
            resample_raster(str(local_raster), str(resampled_path), target_res=target_res)
            local_raster = resampled_path
        
        patch_count = 0
        archive_name = f"dataset_{local_raster.stem}.tar.gz"
        archive_path = tmp_path / archive_name

        with rasterio.open(local_raster) as src:
            src_crs = src.crs
            src_bounds = box(*src.bounds)
            src_nodata = src.nodata if src.nodata is not None else 32767.0
            
            if global_cibles_gdf.crs != src_crs:
                global_cibles_gdf = global_cibles_gdf.to_crs(src_crs)
            cibles_in_zone = global_cibles_gdf[global_cibles_gdf.intersects(src_bounds)]
            
            if cibles_in_zone.empty:
                return 0

            gdf_shifted = shift_cibles_to_anomaly_peak(cibles_in_zone, src)
            pixel_size_m = src.transform[0]
            
            full_image = src.read(1).astype(np.float32)
            nodata_mask = np.isnan(full_image)
            if src_nodata is not None and not np.isnan(src_nodata):
                nodata_mask = nodata_mask | (full_image == src_nodata)
            
            valid_mean = np.mean(full_image[~nodata_mask]) if np.any(~nodata_mask) else 0
            full_image[nodata_mask] = valid_mean
            
            low_freq = gaussian_filter(full_image, sigma=GAUSSIAN_SIGMA)
            full_image_hp = np.clip(full_image - low_freq, -300, 300)

            gdf_A = gdf_shifted[gdf_shifted['famille'] == 'A']
            gdf_B = gdf_shifted[gdf_shifted['famille'] == 'B']
            gdf_C = gdf_shifted[~gdf_shifted['famille'].isin(['A', 'B'])]
            
            # v38 radii (at 0.1m resolution)
            radius_A_px, radius_B_px, radius_C_px = 20, 10, 4
            disks_A = gdf_A.buffer(radius_A_px * pixel_size_m)
            disks_B = gdf_B.buffer(radius_B_px * pixel_size_m)
            disks_C = gdf_C.buffer(radius_C_px * pixel_size_m)
            gdf_disks = gpd.GeoDataFrame(geometry=pd.concat([disks_A, disks_B, disks_C]))
            
            height, width = full_image.shape
            
            with tarfile.open(archive_path, "w:gz") as tar:
                for r in range(0, height, STRIDE):
                    for c in range(0, width, STRIDE):
                        r_end, c_end = min(r + PATCH_SIZE, height), min(c + PATCH_SIZE, width)
                        if np.sum(nodata_mask[r:r_end, c:c_end]) > (PATCH_SIZE * PATCH_SIZE * NODATA_THRESHOLD):
                            continue
                        
                        image_patch = full_image_hp[r:r_end, c:c_end]
                        if image_patch.shape != (PATCH_SIZE, PATCH_SIZE):
                            padded = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
                            padded[:image_patch.shape[0], :image_patch.shape[1]] = image_patch
                            image_patch = padded

                        window = Window(c, r, c_end - c, r_end - r)
                        window_bounds = box(*src.window_bounds(window))
                        disks_in_window = gdf_disks[gdf_disks.intersects(window_bounds)]
                        
                        mask_patch = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
                        if not disks_in_window.empty:
                            mask_unpadded = rasterio.features.rasterize(
                                shapes=[(g, 1) for g in disks_in_window.geometry],
                                out_shape=(window.height, window.width),
                                transform=src.window_transform(window),
                                fill=0, dtype=np.uint8, all_touched=True
                            )
                            mask_patch[:window.height, :window.width] = mask_unpadded

                        patch_name = f"{local_raster.stem}_patch_{patch_count:05d}"
                        
                        img_io = io.BytesIO()
                        np.save(img_io, image_patch.astype(np.float32))
                        img_io.seek(0)
                        tar_info = tarfile.TarInfo(name=f"images/{patch_name}_image.npy")
                        tar_info.size = len(img_io.getbuffer())
                        tar.addfile(tar_info, img_io)
                        
                        mask_io = io.BytesIO()
                        np.save(mask_io, mask_patch.astype(np.float32))
                        mask_io.seek(0)
                        tar_info = tarfile.TarInfo(name=f"masks/{patch_name}_mask.npy")
                        tar_info.size = len(mask_io.getbuffer())
                        tar.addfile(tar_info, mask_io)

                        patch_count += 1

        if patch_count == 0:
            return 0
        upload_blob(str(archive_path), f"gs://{output_bucket}/{output_prefix}/{archive_name}")
        return patch_count

def wrapper(args):
    try:
        return process_zone_and_archive(*args)
    except Exception as e:
        logging.error(f"Error processing {args[0]}: {e}")
        return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-bucket", required=True)
    parser.add_argument("--output-bucket", required=True)
    parser.add_argument("--output-prefix", default="processed_v58_v38_style_stride128")
    parser.add_argument("--raster-prefix", default="raw/raw")
    parser.add_argument("--geojson-path", default="raw/raw/cibles.geojson")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    
    client = storage.Client()
    blobs = list(client.list_blobs(args.input_bucket, prefix=args.raster_prefix))
    raster_blobs = [b.name for b in blobs if b.name.endswith('.tif') and 'zone' in b.name]
    
    # Exclude Zone 19 for validation
    raster_blobs = [b for b in raster_blobs if 'zone_19' not in b]
    
    if not raster_blobs:
        logging.error("No rasters found.")
        sys.exit(1)
        
    with tempfile.NamedTemporaryFile(suffix=".geojson") as tmp:
        download_blob(f"gs://{args.input_bucket}/{args.geojson_path}", tmp.name)
        global_cibles = gpd.read_file(tmp.name)
        
    tasks = [(b, args.input_bucket, global_cibles, args.output_bucket, args.output_prefix) for b in raster_blobs]
    
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        results = list(tqdm(exe.map(wrapper, tasks), total=len(tasks)))
        
    logging.info(f"Total patches generated: {sum(results)}")
