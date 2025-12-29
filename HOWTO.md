# release/v62-Ultimate HOWTO: Recreating the 96% Recall Ensemble

This guide details the step-by-step process to reproduce the **Ultimate 5-Way Ensemble** for UXO detection. This ensemble achieves a record-breaking **96.2% Recall** on Zone 19 by fusing predictions from Swin Transformer, U-Net, HRNet, SegFormer, and YOLO11.

---

## 1. Directory Structure

-   `scripts/swin/`: **Swin Transformer** (Tiny) training & inference code.
-   `scripts/unet/`: **Stabilized U-Net** training code (High effective batch size).
-   `scripts/hrnet/`: **High-Resolution Net** (Spatial precision).
-   `scripts/segformer/`: **Transformer-based Segmentation**.
-   `scripts/yolo/`: **Discrete Object Detection** (YOLO11).
-   `ensemble/`: **Fusion Logic** (`ensemble_5way.py`) to combine predictions.
-   `weights/`: Pre-trained model weights (Checkpoints).
-   `results/`: Benchmarks and evaluation metrics.

---

## 2. Prerequisites

Ensure your environment has the following core dependencies installed:
-   **Python 3.10+**
-   **PyTorch 2.0+** (with CUDA support)
-   **Gdal / Rasterio** (Geospatial processing)
-   **Ultralytics** (for YOLO11)
-   **Scikit-image** & **Scipy** (Post-processing)

```bash
pip install torch torchvision rasterio geopandas pandas shapely scikit-image scipy ultralytics tqdm google-cloud-storage
```

---

## 3. Reproduction Workflow

### Step 1: Data Preparation
Each model directory contains a `create_dataset.py` script. This script typically downloads partitioned data shards (`.tar.gz`) from Cloud Storage and extracts them into a local `images/` and `masks/` structure suitable for training.

*Note: The models expect data in a specific "Label Shifted" format (Inverse RTP).*

### Step 2: Training Evaluation Models
To retrain the models from scratch, execute the `train.py` script in each subdirectory.

#### A. Swin Transformer (Generalist)
```bash
python3 scripts/swin/train.py \
    --data-bucket gs://your-data-bucket/swin-prep \
    --checkpoint-dir runs/swin \
    --epochs 200 --batch-size 128
```

#### B. HRNet (Spatial Precision)
Uses **AdaptiveWingLoss** for precise peak localization.
```bash
python3 scripts/hrnet/train.py \
    --data-bucket gs://your-data-bucket/hrnet-prep \
    --checkpoint-dir runs/hrnet \
    --epochs 500 --batch-size 128
```

#### C. SegFormer (Context)
```bash
python3 scripts/segformer/train.py \
    --data-bucket gs://your-data-bucket/segformer-prep \
    --checkpoint-dir runs/segformer \
    --epochs 500 --batch-size 128
```

#### D. YOLO11 (Object Detection)
Requires `imgsz` matching your patch size (e.g., 256).
```bash
python3 scripts/yolo/train.py \
    --data-bucket gs://your-data-bucket/yolo-prep \
    --checkpoint-dir runs/yolo \
    --imgsz 256 --epochs 100
```

### Step 3: Inference (Generating Predictions)
Generate the specific heatmaps (or GeoJSONs for YOLO) required for the ensemble.

**Standard Segmentation Models (Swin, U-Net, HRNet, SegFormer):**
```bash
python3 scripts/hrnet/predict.py \
    --input-raster data/raw/zone_19_mag.tif \
    --output-raster results/inference/hrnet_heatmap.tif \
    --model weights/hrnet_best.pth
```
*(Repeat for Swin, U-Net, and SegFormer using their respective scripts and weights)*

**YOLO (Object Detection):**
Use the YOLO `predict.py` to generate a GeoJSON of bounding boxes/points.
```bash
python3 scripts/yolo/predict.py \
    --input-raster data/raw/zone_19_mag.tif \
    --output-geojson results/inference/yolo_preds.geojson \
    --model weights/yolo_best.pt
```

### Step 4: 5-Way Ultimate Fusion
The final step combines the 5 predictions using **Weighted Box Fusion (WBF)**.

**Critical**: Ensure the following files exist at these locations (or update `ensemble_5way.py` paths):
1.  `scripts/ensemble/zone_19_swin_heatmap.tif`
2.  `scripts/unet/zone_19_heatmap.tif`
3.  `results/inference/hrnet_heatmap.tif`
4.  `results/inference/segformer_heatmap.tif`
5.  `results/inference/yolo_preds.geojson`

**Run the Ensemble:**
```bash
python3 ensemble/ensemble_5way.py \
    --dist 3.0 \
    --output release/v62-Ultimate/results/ultimate_fused.geojson
```
*`--dist 3.0` sets the clustering distance to 3.0 meters (Operational safety margin).*

---

## 4. Evaluation & Results
The resulting `ultimate_fused.geojson` contains the final target list. 

**v62-Ultimate Performance on Zone 19:**
| Metric | 1m Buffer | 3m Buffer |
| :--- | :--- | :--- |
| **Recall** | 89.42% | **96.22%** |
| **Precision** | 68.10% | 78.90% |

Validation is performed using `v62_run_evals.py` (if available) or standard geo-spatial comparison tools against ground truth.

---

*For detailed scientific background, please refer to the main [README.md](./README.md).*
