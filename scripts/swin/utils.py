import os
import re
import logging
import rasterio
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
from google.cloud import storage
from skimage.feature import peak_local_max
from shapely.geometry import Point

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_gcs_url(url):
    """Parses gs://bucket/key into (bucket, key)."""
    match = re.match(r"gs://([^/]+)/(.+)", url)
    if not match:
        return None, None
    return match.group(1), match.group(2)

def download_blob(gs_path, local_path):
    """Downloads a GCS blob to a local path."""
    bucket_name, blob_name = parse_gcs_url(gs_path)
    if not bucket_name:
        logging.error(f"Invalid GCS URL: {gs_path}")
        return False
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    logging.info(f"Downloaded {gs_path} to {local_path}")
    return True

def upload_blob(local_path, gs_path):
    """Uploads a local file to a GCS path."""
    bucket_name, blob_name = parse_gcs_url(gs_path)
    if not bucket_name:
        logging.error(f"Invalid GCS URL: {gs_path}")
        return False
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    logging.info(f"Uploaded {local_path} to {gs_path}")
    return True

def resample_raster(input_path, output_path, target_res=0.10):
    """Resamples a raster to a target resolution while preserving CRS."""
    logging.info(f"Resampling {input_path} to {target_res}m...")
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, src.crs, src.width, src.height, *src.bounds, 
            resolution=(target_res, target_res)
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': src.crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=src.crs,
                    resampling=Resampling.bilinear
                )
    logging.info(f"Saved resampled raster to {output_path}")
    return output_path

def extract_peaks(heatmap_path, threshold=0.05, min_distance=60):
    """Extracts local peaks from a heatmap raster."""
    with rasterio.open(heatmap_path) as src:
        heatmap_data = src.read(1)
        src_crs = src.crs
        src_transform = src.transform
        
        coordinates = peak_local_max(
            heatmap_data,
            min_distance=min_distance,
            threshold_abs=threshold
        )
        
        if coordinates.size == 0:
            return None

        peak_rows = coordinates[:, 0]
        peak_cols = coordinates[:, 1]
        confidences = heatmap_data[peak_rows, peak_cols]
        xs, ys = src_transform * (peak_cols, peak_rows)
        centroids = [Point(x, y) for x, y in zip(xs, ys)]
        
        import geopandas as gpd
        return gpd.GeoDataFrame({'geometry': centroids, 'confidence': confidences}, crs=src_crs)
