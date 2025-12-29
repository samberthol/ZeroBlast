import rasterio
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import box
import os

def generate_overlay_visualization(raster_path, cibles_path, predictions_path, output_path, title='Zone 19 Performance Overlay (v60)'):
    print(f"Reading raster: {raster_path}")
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        nodata = src.nodata
        bounds = box(*src.bounds)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        crs = src.crs

    # Masking and robust scaling for high contrast
    mask = (data == nodata) | np.isnan(data)
    valid_data = data[~mask]
    masked_data = np.ma.masked_where(mask, data)
    vmin, vmax = np.percentile(valid_data, 2), np.percentile(valid_data, 98)
    
    # Load Predictions
    print(f"Reading predictions: {predictions_path}")
    preds = gpd.read_file(predictions_path)
    if not preds.empty and preds.crs != crs:
        preds = preds.to_crs(crs)

    # Load Ground Truth
    print(f"Reading targets: {cibles_path}")
    all_truth = gpd.read_file(cibles_path)
    if all_truth.crs != crs:
        all_truth = all_truth.to_crs(crs)
    zone_targets = all_truth[all_truth.intersects(bounds)]
    
    print("Generating high-res overlay plot...")
    # Increased figure size and DPI for high-res requirement
    fig, ax = plt.subplots(figsize=(30, 30))
    im = ax.imshow(masked_data, extent=extent, cmap='magma', vmin=vmin, vmax=vmax)
    
    # Plot Ground Truth (Cyan Circles) - Increased size for visibility
    if not zone_targets.empty:
        zone_targets.plot(ax=ax, color='cyan', marker='o', markersize=150, label=f'Ground Truth ({len(zone_targets)})', facecolor='none', edgecolor='cyan', linewidth=2.0)
    
    # Plot Predictions (Lime Green Dots) - Precision mode
    if not preds.empty:
        preds.plot(ax=ax, color='lime', marker='.', markersize=30, alpha=0.9, label=f'v60 Predictions ({len(preds)})')

    ax.set_title(title, fontsize=24, fontweight='bold')
    fig.colorbar(im, ax=ax, label='nT', fraction=0.046, pad=0.04)
    ax.set_xlabel('Easting (m)', fontsize=14)
    ax.set_ylabel('Northing (m)', fontsize=14)
    ax.legend(loc='upper right', fontsize=16)
    
    plt.tight_layout()
    # High resolution save
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"Success! High-res visualization saved to: {output_path}")

if __name__ == "__main__":
    raster = "../../zone_19.tif"
    cibles = "../../scripts/v57/cibles.geojson"
    # Using the 50% threshold predictions obtained previously
    predictions = "preds/preds_t50.geojson"
    output = "v60_zone19_viz_highres.png"
    
    if os.path.exists(predictions):
        generate_overlay_visualization(raster, cibles, predictions, output)
    else:
        print(f"Error: {predictions} not found. Run extract_predictions.py first.")
