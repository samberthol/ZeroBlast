import os
import argparse
import logging
import rasterio
import numpy as np
import geopandas as gpd
from skimage.feature import peak_local_max
from shapely.geometry import Point, box
from tqdm import tqdm
import pandas as pd
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_peaks(heatmap_path, threshold, min_distance=10):
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
    """
    Applies spatial Non-Maximum Suppression to a GeoDataFrame of points.
    """
    if gdf is None or len(gdf) == 0:
        return gdf
        
    # Extract coordinates
    coords = np.array([(p.x, p.y) for p in gdf.geometry])
    confidences = gdf['confidence'].values
    
    # Sort by confidence descending
    idx_sorted = np.argsort(-confidences)
    coords = coords[idx_sorted]
    confidences = confidences[idx_sorted]
    original_indices = gdf.index[idx_sorted]
    
    tree = cKDTree(coords)
    keep = np.ones(len(coords), dtype=bool)
    
    for i in range(len(coords)):
        if keep[i]:
            # Find all points within distance_threshold
            neighbors = tree.query_ball_point(coords[i], distance_threshold)
            # Mark all neighbors (except self) as suppressed
            for n in neighbors:
                if n > i:
                    keep[n] = False
    
    return gdf.loc[original_indices[keep]].copy()

def evaluate(truth_gdf, pred_gdf, zone_bounds, buffer_distance_m):
    # Filter to zone
    t_gdf = truth_gdf[truth_gdf.intersects(zone_bounds)].copy()
    p_gdf = pred_gdf[pred_gdf.intersects(zone_bounds)].copy()
    
    n_truth = len(t_gdf)
    n_pred = len(p_gdf)
    
    if n_truth == 0:
        return 0, 0, 0, 0, 0

    truth_buffers = t_gdf.buffer(buffer_distance_m)
    true_positives_gdf = gpd.sjoin(p_gdf, gpd.GeoDataFrame(geometry=truth_buffers, crs=t_gdf.crs), how='left', predicate='intersects')
    
    tp_mask = true_positives_gdf['index_right'].notna()
    detected_truth_targets = true_positives_gdf['index_right'].nunique()
    false_positives = (true_positives_gdf['index_right'].isna()).sum()
    false_negatives = n_truth - detected_truth_targets
    true_positive_predictions = tp_mask.sum()

    precision = true_positive_predictions / (true_positive_predictions + false_positives) if (true_positive_predictions + false_positives) > 0 else 0
    recall = detected_truth_targets / (detected_truth_targets + false_negatives) if (detected_truth_targets + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1, n_truth, n_pred

def main():
    results_dir = "results/v62/inference"
    truth_path = "data/raw/cibles.geojson"
    raster_path = "data/raw/zone_19.tif"
    
    thresholds = [0.1, 0.2, 0.3, 0.5, 0.8, 0.9]
    buffers = [1.0, 3.0]
    models = ["swin", "unet", "hrnet", "segformer", "yolo", "ultimate_3m", "ultimate_1m"]
    
    logging.info("Loading ground truth and zone bounds...")
    truth_gdf = gpd.read_file(truth_path)
    with rasterio.open(raster_path) as src:
        zone_bounds = box(*src.bounds)
        if truth_gdf.crs != src.crs:
            truth_gdf = truth_gdf.to_crs(src.crs)

    results = []

    for model in models:
        logging.info(f"Evaluating {model}...")
        
        if model == "swin":
            heatmap_path = "scripts/ensemble/zone_19_swin_heatmap.tif"
        elif model == "unet":
            heatmap_path = "scripts/unet/zone_19_heatmap.tif"
        elif model == "yolo":
            pred_path = os.path.join(results_dir, "yolo_preds.geojson")
            full_pred_gdf = gpd.read_file(pred_path)
            if full_pred_gdf.crs != truth_gdf.crs:
                full_pred_gdf = full_pred_gdf.to_crs(truth_gdf.crs)
        elif model == "ultimate_3m":
            pred_path = "results/ultimate_fused.geojson"
            full_pred_gdf = gpd.read_file(pred_path)
            if full_pred_gdf.crs != truth_gdf.crs:
                full_pred_gdf = full_pred_gdf.to_crs(truth_gdf.crs)
        elif model == "ultimate_1m":
            # Assuming 1m file exists if previously run, otherwise handle or default
            pred_path = "results/ultimate_fused_1m.geojson" 
            if os.path.exists(pred_path):
                full_pred_gdf = gpd.read_file(pred_path)
                if full_pred_gdf.crs != truth_gdf.crs:
                    full_pred_gdf = full_pred_gdf.to_crs(truth_gdf.crs)
            else:
                logging.warning(f"{pred_path} not found, skipping 1m evaluation.")
                continue
        else:
            heatmap_path = os.path.join(results_dir, f"{model}_heatmap.tif")

        for t in thresholds:
            if model == "yolo":
                pred_gdf = full_pred_gdf[full_pred_gdf['confidence'] >= t].copy()
                # Apply Global NMS to YOLO predictions
                pred_gdf = spatial_nms(pred_gdf, distance_threshold=2.0)
            elif model == "ultimate_3m":
                pred_gdf = full_pred_gdf[full_pred_gdf['confidence'] >= t].copy()
            elif model == "ultimate_1m":
                pred_gdf = full_pred_gdf[full_pred_gdf['confidence'] >= t].copy()
            else:
                pred_gdf = extract_peaks(heatmap_path, threshold=t)
            
            if pred_gdf is None or len(pred_gdf) == 0:
                for b in buffers:
                    results.append([model, t, b, 0, 0, 0, 0, 0])
                continue

            for b in buffers:
                p, r, f1, nt, npred = evaluate(truth_gdf, pred_gdf, zone_bounds, b)
                results.append([model, t, b, p, r, f1, nt, npred])

    df = pd.DataFrame(results, columns=["Model", "Threshold", "Buffer", "Precision", "Recall", "F1", "N_Truth", "N_Pred"])
    df.to_csv("results/v62/eval_results.csv", index=False)
    
    # Generate Table
    for b in buffers:
        print(f"\n### Results for Buffer {b}m")
        subset = df[df['Buffer'] == b].copy()
        # Pivot for easier comparison
        pivot = subset.pivot(index="Threshold", columns="Model", values=["Precision", "Recall", "F1"])
        print(pivot.to_markdown())

if __name__ == "__main__":
    main()
