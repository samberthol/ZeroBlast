import os
import argparse
import logging
import rasterio
import numpy as np
from skimage.feature import peak_local_max
from shapely.geometry import Point
import geopandas as gpd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_peaks(heatmap_path, threshold=0.05, min_distance=10):
    """Extracts local peaks from a heatmap raster."""
    with rasterio.open(heatmap_path) as src:
        heatmap_data = src.read(1)
        src_crs = src.crs
        src_transform = src.transform
        
        # Replace NaNs or nodata with 0 for peak detection
        heatmap_data = np.nan_to_num(heatmap_data, nan=0.0, posinf=0.0, neginf=0.0)
        heatmap_data[heatmap_data < 0] = 0
        
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
        
        return gpd.GeoDataFrame({'geometry': centroids, 'confidence': confidences}, crs=src_crs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heatmap", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    thresholds = [0.9, 0.8, 0.5, 0.3, 0.2, 0.1]
    
    for t in thresholds:
        logging.info(f"Extracting peaks for threshold {t}...")
        gdf = extract_peaks(args.heatmap, threshold=t, min_distance=10)
        
        if gdf is not None:
            output_path = os.path.join(args.output_dir, f"preds_t{int(t*100)}.geojson")
            gdf.to_file(output_path, driver='GeoJSON')
            logging.info(f"Saved {len(gdf)} predictions to {output_path}")
        else:
            logging.warning(f"No peaks found for threshold {t}")

if __name__ == "__main__":
    main()
