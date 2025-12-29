# Project ZeroBlast: Ultimate UXO Detection System

## 1. Project Overview

**Project ZeroBlast** is a state-of-the-art machine learning initiative designed to automate the detection of Unexploded Ordnance (UXO) from geophysical survey data. 

The primary objective is to process high-resolution magnetometry rasters and identify magnetic anomalies corresponding to buried ferromagnetic targets. The **Ultimate** release represents the culmination of this research, delivering a 5-model ensemble that achieves a safety-critical **96.2% Recall** rate, radically outperforming traditional signal processing methods.

---

## 2. Scientific Methodology

The core scientific challenge in UXO detection is the "Dipole Effect"—magnetic targets appear as complex positive/negative dipoles rather than simple dots, and their appearance changes based on latitude and depth.

### 2.1 The "Label Shifting" Innovation
Early experiments proved that standard segmentation failed because the "True" GPS location of a target often lies in the low-gradient zero-crossing between the magnetic poles. 

We developed **Label Shifting (Inverse RTP)**, a physics-informed strategy where we calculate the theoretical magnetic peak for each target using the dipole equation and shift the training label to align with this peak. This transforms the problem from "guess the hidden center" to "detect the visible anomaly," resulting in a robust, learnable task.

### 2.2 The "Ensemble" Hypothesis
No single model architecture is perfect for all anomaly types (which range from tiny 1m surface clutter to deep 500kg bombs). The Ultimate system relies on **Architectural Diversity**:
- **Transformers** (Swin, SegFormer) excel at capturing global context and filtering noise.
- **CNNS** (HRNet, U-Net) excel at precise, pixel-perfect spatial localization.
- **Object Detectors** (YOLO) excel at learning discrete object signatures.

By fusing these distinct "expert opinions," we achieve performance greater than the sum of its parts.

---

## 3. Source Data

The system is trained and evaluated on two primary data streams:
1.  **Magnetometry Rasters (TMI)**: High-resolution Total Magnetic Intensity maps (GeoTIFF).
    -   *Preprocessing*: Gaussian High-Pass Filter (removes geological trends).
    -   *Resolution*: 0.10m to 0.20m per pixel.
2.  **Ground Truth Vectors**: GPS coordinates of physically verified targets.

---

## 4. The Ultimate Ensemble (5-Way)

The system fuses predictions from five distinct models, selected via a rigorous "Tournament" process.

| Model | Architecture | Role / Strength |
| :--- | :--- | :--- |
| **Swin** | **Swin Transformer (Tiny)** | **The Generalist.** Uses shifted window attention to capture mid-range dependencies. Excellent balance of precision and recall. |
| **U-Net** | **Stabilized U-Net** | **The Safety Net.** Trained with a massive effective batch size (512) to be extremely sensitive. Highest individual recall. |
| **HRNet** | **HRNet (High-Res Net)** | **The Sniper.** Maintains high-resolution representations throughout the network. Provides the most spatially precise peak localization. |
| **SegFormer** | **SegFormer (B0)** | **The Context Awareness.** A lightweight transformer that excels at distinguishing complex geological noise from true signals. |
| **YOLO** | **YOLO11** | **The Object Detector.** A distinct regression paradigm that "looks" for discrete objects rather than segmenting pixels. Adds crucial diversity. |

**Fusion Mechanism**: The predictions are combined using **Weighted Box Fusion (WBF)** with a customized weighting strategy that prioritizes high-confidence agreement while preserving solitary high-recall detections.

---

## 5. Performance Results (Zone 19)

The Ultimate Ensemble was benchmarked on "Zone 19," a rigorous hold-out region containing complex noise and varied target types.

### Primary Metrics (3m Buffer)
-   **Recall**: **96.22%** (New SOTA)
-   **Precision**: 78.9%
-   **F1-Score**: 0.867

### Precision Breakthrough (1m Buffer)
Historically, models struggled to pinpoint targets within 1 meter. The ensemble shatters this ceiling:
-   **Recall @ 1m**: **89.4%**
-   **F1 @ 1m**: 0.705

*Note: A 3m buffer is the standard operational safety margin for remediation excavation.*

---

## 6. Repository Structure

This release package contains everything needed to reproduce these results:

```
release/v62-Ultimate/
├── scripts/
│   ├── swin/       # Swin Transformer Training Code
│   ├── unet/       # Stabilized U-Net Training Code
│   ├── hrnet/      # High-Resolution Net Training Code
│   ├── segformer/  # SegFormer Training Code
│   └── yolo/       # YOLO11 Training Code
├── ensemble/       # Fusion Logic (WBF) & Evaluation Scripts
├── weights/        # Pre-trained Model Checkpoints (x5)
└── results/        # Detailed CSV Benchmarks & Visualizations
```

For detailed reproduction steps, please refer to [HOWTO.md](./HOWTO.md).

---

*Project ZeroBlast - Advanced Machine Learning for Humanitarian De-mining*
