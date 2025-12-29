# Project ZeroBlast: Ultimate UXO Detection System

## 1. Project Overview

**Project ZeroBlast** is a machine learning initiative designed to automate the detection of Unexploded Ordnance (UXO) from geophysical survey data. 

The primary objective is to process high-resolution magnetometry rasters and identify magnetic anomalies corresponding to buried ferromagnetic targets. The **Ultimate** release represents the final iteration of this research, delivering a 5-model ensemble that achieves a safety-oriented **96.2% Recall** rate, outperforming traditional signal processing methods.

> **Data Confidentiality & Model Weights**: The data used to train these models is confidential and restricted. Consequently, the pre-trained model weights cannot be shared publicly. This repository is intended solely to share the **scientific methodology, architectural breakthroughs, and reproduction scripts** developed during the project.

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

### 2.3 The Signal-to-Noise Challenge
Magnetometry data is inherently "dirty." Geological variations (e.g., magnetic rocks) and sensor drift create a complex background noise floor. **Famille C** targets (tiny UXOs or deep objects) produce signals in the 50-500 nT range, which is often indistinguishable from this background noise. The system must learn to recognize the *geometric* fingerprint of a dipole rather than relying on raw intensity.

### 2.4 Synthetic Noise Injection (Fractal Noise)
To improve robustness, we inject synthetic noise during training. While Swin and UNet use standard Gaussian noise, **HRNet** utilizes **Fractal (1/f^α) Noise**. Unlike white noise, Fractal Noise has spatial correlations that mimic real-world geological clutter, forcing the model to differentiate between "structured noise" and "structured signals."

### 2.5 Remanent Magnetization & Rotations
UXOs are rarely aligned perfectly with the Earth's magnetic field. This "Remanent Magnetization" changes the dipole's appearance. 
- **Randomized Dipole Orientation**: During data preparation, we simulate targets with random magnetic moments (randomized $m_x, m_y, m_z$).
- **Geometric Invariance**: Training scripts apply horizontal flip augmentations to ensure the models are invariant to the survey direction.

### 2.6 TMI Clipping & Signal Normalization
Raw magnetometry data often contains extreme spikes from surface-level metallic clutter or sensor errors that can exceed 10,000 nT. To prevent these outliers from dominating the training loss, we implement a strict **Clipping Strategy**:
- **Range**: All TMI values are clipped to the $[-300, 300]$ nT range.
- **Rationale**: Most UXO signatures of interest fall within this window. Clipping preserves the signature of deep/small targets while "capping" the influence of massive surface objects.

### 2.7 Advanced Architectural Particularities
Several specific techniques were implemented to reach the 96% recall ceiling:
- **Adaptive Wing Loss (AWing)**: Used in HRNet to prioritize spatial precision in heatmap regression by focusing on "near-peak" pixels.
- **Deep Supervision**: Swin and HRNet utilize multi-scale loss calculations (calculating loss at intermediate layers) to force the network to learn meaningful features early in the processing chain.
- **16-bit to 8-bit Transform**: For YOLOv11 training, we implement a linear scaling of the clipped data to 8-bit 0-255 images, enabling the use of pre-trained detection weights without losing critical anomaly signatures.

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

**Fusion Mechanism**: The Ultimate Ensemble utilizes **Weighted Box Fusion (WBF)** rather than standard NMS. Instead of discarding overlapping detections, WBF calculates a confidence-weighted average of the results from all five models, converging on a "Consensus Centroid." This mathematical smoothing is the primary driver behind our record-breaking **89.4% Recall @ 1m**, as it effectively eliminates individual model spatial offsets.

![Ultimate Ensemble Performance Overlay](file:///usr/local/google/home/sberthollier/Code/pyro-remediation/release/v62-Ultimate/results/ultimate_ensemble_performance.png)
*Figure 1: Ultimate Ensemble predictions overlaid on TMI data, showcasing the high-density recall achieved via multi-model fusion.*

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
└── results/        # Detailed CSV Benchmarks & Visualizations
```

For detailed reproduction steps, please refer to [HOWTO.md](./HOWTO.md).

---

*Project ZeroBlast - Advanced Machine Learning for Humanitarian De-mining*
