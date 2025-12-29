# release/v62-Ultimate HOWTO: Recreating the 96% Recall Ensemble

This guide details the step-by-step process to reproduce the **Ultimate 5-Way Ensemble** for UXO detection. This ensemble achieves a record-breaking **96.2% Recall** on Zone 19 by fusing predictions from Swin Transformer, U-Net, HRNet, SegFormer, and YOLO11.

---

## 1. Directory Structure

-   `scripts/swin/`: **Swin Transformer** (Tiny) training & inference code.
-   `scripts/unet/`: **Stabilized U-Net** training code (High effective batch size).
-   `scripts/hrnet/`: **High-Resolution Net** (Spatial precision).
-   `scripts/segformer/`: **Transformer-based Segmentation**.
-   `scripts/yolo/`: **Discrete Object Detection** (YOLO11).
-   `scripts/ensemble/`: **Fusion Logic** (`ensemble_5way.py`) to combine predictions.
-   `results/`: Benchmarks and evaluation metrics.

---

## 2. Prerequisites

Ensure your environment has the following core dependencies installed:
-   **Python 3.10+**
-   **PyTorch 2.3+** (with CUDA 12.1+ support)
-   **Gdal / Rasterio** (Geospatial processing)
-   **Timm** & **Einops** (Backbone and tensor manipulations)
-   **Transformers** & **Huggingface-hub** (for SegFormer)
-   **Ultralytics** (for YOLO11)
-   **OpenCV-Python** & **Pillow** (Image processing for detection)
-   **Scikit-image**, **Scipy**, **Scikit-learn** (Post-processing and metrics)
-   **Google-cloud-storage** (Data synchronization)

```bash
pip install torch torchvision rasterio geopandas pandas shapely \
            scikit-image scipy scikit-learn ultralytics tqdm \
            google-cloud-storage timm einops transformers \
            huggingface-hub opencv-python pillow
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
    --model <path/to/your/hrnet_best.pth>
```
*(Repeat for Swin, U-Net, and SegFormer, ensuring output filenames match `swin_heatmap.tif`, `unet_heatmap.tif`, and `segformer_heatmap.tif` in the `results/inference/` directory)*

**YOLO (Object Detection):**
Use the YOLO `predict.py` to generate a GeoJSON of bounding boxes/points.
```bash
python3 scripts/yolo/predict.py \
    --input-raster data/raw/zone_19_mag.tif \
    --output-geojson results/inference/yolo_preds.geojson \
    --model <path/to/your/yolo_best.pt>
```

### Step 4: 5-Way Ultimate Fusion
The final step combines the 5 predictions using **Weighted Box Fusion (WBF)**.

**Run the Ensemble:**
```bash
python3 scripts/ensemble/ensemble_5way.py \
    --swin results/inference/swin_heatmap.tif \
    --unet results/inference/unet_heatmap.tif \
    --hrnet results/inference/hrnet_heatmap.tif \
    --segformer results/inference/segformer_heatmap.tif \
    --yolo results/inference/yolo_preds.geojson \
    --dist 3.0 \
    --output results/ultimate_fused.geojson
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
