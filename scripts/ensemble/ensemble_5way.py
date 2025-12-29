import os
import argparse
import logging
import rasterio
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from skimage.feature import peak_local_max
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_peaks(heatmap_path, threshold, min_distance=10):
    if not os.path.exists(heatmap_path):
        logging.error(f"Heatmap not found: {heatmap_path}")
        return None
    with rasterio.open(heatmap_path) as src:
        heatmap_data = src.read(1)
        src_crs = src.crs
        src_transform = src.transform
        
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

def spatial_nms(gdf, distance_threshold=2.0):
    if gdf is None or len(gdf) == 0:
        return gdf
    coords = np.array([(p.x, p.y) for p in gdf.geometry])
    confidences = gdf['confidence'].values
    idx_sorted = np.argsort(-confidences)
    coords = coords[idx_sorted]
    original_indices = gdf.index[idx_sorted]
    tree = cKDTree(coords)
    keep = np.ones(len(coords), dtype=bool)
    for i in range(len(coords)):
        if keep[i]:
            neighbors = tree.query_ball_point(coords[i], distance_threshold)
            for n in neighbors:
                if n > i:
                    keep[n] = False
    return gdf.loc[original_indices[keep]].copy()

def weighted_box_fusion_5(gdfs, names, weights, distance_threshold=3.0):
    """
    Weighted Box Fusion for 5 point-based models.
    """
    all_points = []
    for gdf, name, weight in zip(gdfs, names, weights):
        if gdf is None or gdf.empty:
            continue
        df = gdf.copy()
        df['source'] = name
        df['model_weight'] = weight
        all_points.append(df)
        
    if not all_points:
        return None
        
    combined = pd.concat(all_points).reset_index(drop=True)
    coords = np.array([[p.x, p.y] for p in combined.geometry])
    tree = cKDTree(coords)
    
    used = np.zeros(len(combined), dtype=bool)
    fused_results = []
    
    for i in range(len(combined)):
        if used[i]:
            continue
            
        cluster_indices = tree.query_ball_point(coords[i], distance_threshold)
        cluster_indices = [idx for idx in cluster_indices if not used[idx]]
        if not cluster_indices:
            continue
            
        used[cluster_indices] = True
        cluster = combined.iloc[cluster_indices].copy()
        
        # Weighted coordinates
        combined_weight = cluster['confidence'].values * cluster['model_weight'].values
        sum_weights = combined_weight.sum()
        
        if sum_weights > 0:
            avg_x = (cluster.geometry.x * combined_weight).sum() / sum_weights
            avg_y = (cluster.geometry.y * combined_weight).sum() / sum_weights
        else:
            avg_x, avg_y = coords[i]
            
        # Confidence fusion with consensus boost
        max_conf = cluster['confidence'].max()
        unique_models = cluster['source'].nunique()
        
        # Consensus boost: +5% per additional model agreeing
        boost = 1.0 + (unique_models - 1) * 0.05
        fused_conf = min(1.0, max_conf * boost)
            
        fused_results.append({
            'geometry': Point(avg_x, avg_y),
            'confidence': fused_conf,
            'consensus_count': unique_models,
            'models': ",".join(cluster['source'].unique())
        })
        
    return gpd.GeoDataFrame(fused_results, crs=combined.crs)

def main():
    parser = argparse.ArgumentParser(description="v60+v62 5-model point-based ensemble")
    parser.add_argument("--swin", help="Path to Swin heatmap .tif")
    parser.add_argument("--unet", help="Path to UNet heatmap .tif")
    parser.add_argument("--hrnet", help="Path to HRNet heatmap .tif")
    parser.add_argument("--segformer", help="Path to SegFormer heatmap .tif")
    parser.add_argument("--yolo", help="Path to YOLO predictions .geojson")
    parser.add_argument("--dist", type=float, default=3.0, help="Distance threshold for fusion (meters)")
    parser.add_argument("--output", default="results/v62-Ultimate/results/ultimate_fused.geojson", help="Output path")
    args = parser.parse_args()

    # Paths (Default fallbacks if not provided)
    swin_h = args.swin or "results/inference/swin_heatmap.tif"
    unet_h = args.unet or "results/inference/unet_heatmap.tif"
    hrnet_h = args.hrnet or "results/inference/hrnet_heatmap.tif"
    segformer_h = args.segformer or "results/inference/segformer_heatmap.tif"
    yolo_g = args.yolo or "results/inference/yolo_preds.geojson"
    
    logging.info(f"5-Way Fusion Distance: {args.dist}m")
    
    logging.info(f"Extracting Swin peaks (from {swin_h}, threshold=0.20)...")
    gdf_swin = extract_peaks(swin_h, threshold=0.20)
    
    logging.info(f"Extracting UNet peaks (from {unet_h}, threshold=0.30)...")
    gdf_unet = extract_peaks(unet_h, threshold=0.30)
    
    logging.info(f"Extracting HRNet peaks (from {hrnet_h}, threshold=0.10)...")
    gdf_hrnet = extract_peaks(hrnet_h, threshold=0.10)
    
    logging.info(f"Extracting SegFormer peaks (from {segformer_h}, threshold=0.10)...")
    gdf_segformer = extract_peaks(segformer_h, threshold=0.10)
    
    if os.path.exists(yolo_g):
        logging.info(f"Loading YOLO detections (from {yolo_g})...")
        gdf_yolo = gpd.read_file(yolo_g)
        gdf_yolo = spatial_nms(gdf_yolo, distance_threshold=2.0)
    else:
        logging.warning(f"YOLO detections not found at {yolo_g}")
        gdf_yolo = None
    
    gdfs = [gdf_swin, gdf_unet, gdf_hrnet, gdf_segformer, gdf_yolo]
    names = ["swin", "unet", "hrnet", "segformer", "yolo"]
    weights = [1.0, 1.0, 1.0, 1.1, 1.0] # Slightly favor SegFormer's modern recall
    
    logging.info(f"Running 5-model Weighted Box Fusion (dist={args.dist})...")
    fused_gdf = weighted_box_fusion_5(gdfs, names, weights, distance_threshold=args.dist)
    
    if fused_gdf is not None:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        fused_gdf.to_file(args.output, driver='GeoJSON')
        logging.info(f"Saved {len(fused_gdf)} fused targets to {args.output}")
    else:
        logging.error("Fusion failed - no points found.")

if __name__ == "__main__":
    main()
