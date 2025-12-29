# Model Benchmark Results: Zone 19

This document archives the evaluation results for the **HRNet**, **SegFormer**, **YOLO**, and **Ultimate Ensemble** on Zone 19.

## Evaluation Configuration
- **Zones**: Training (Various), Evaluation (Zone 19)
- **Thresholds Swept**: 10%, 20%, 30%, 50%, 80%, 90%
- **Buffers**: 1m, 3m
- **Deduplication**: Global NMS (2m radius) applied to YOLO.
- **Ensemble Strategy**: 
  - **Ultimate 5-Way**: WBF of Swin, UNet, HRNet, SegFormer, YOLO.

## ULTIMATE RESULT: 95% Recall Ceiling Shattered
The **Ultimate 5-Way Ensemble** achieved a record-breaking **96.22% Recall** at 10% threshold (3m buffer) and nearly hit **90% Recall at 1m**.

## Comparative Results (3m Buffer)

| Model | Threshold | Precision | Recall | F1-Score | Target Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ultimate Ensemble** (1m fusion) | 10% | 0.789 | **0.962** | 0.867 | 2,245 |
| **Ultimate Ensemble** (3m fusion) | 10% | 0.756 | **0.957** | 0.845 | 1,568 |
| **UNet (v60)** | 10% | 0.114 | **0.998** | 0.205 | 21,768 |
| **SegFormer** | 10% | 0.838 | 0.873 | 0.855 | 1,454 |
| **YOLO** | 10% | 0.884 | 0.841 | **0.862** | 1,295 |
| **HRNet** | 10% | **0.921** | 0.806 | 0.860 | 1,275 |

## 1m Buffer Breakthrough (Safety Ceiling)

| Model | Threshold | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Ultimate Ensemble** (1m fusion) | 10% | 0.582 | **0.894** | 0.705 |
| **v61-Zero-Miss** | 10% | 0.286 | 0.672 | 0.401 |

## Prediction Counts (Number of Targets)

| Threshold | HRNet | SegFormer | YOLO | Swin (v58) | UNet (v60) | Ultimate (Fused) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **10%** | 1,275 | 1,454 | 1,295 | 24,872 | 21,768 | 1,568 |
| **20%** | 1,045 | 1,240 | 939 | 1,155 | 17,821 | 1,430 |
| **30%** | 920 | 1,078 | 674 | 958 | 1,388 | 1,301 |
| **50%** | 827 | 877 | 89 | 743 | 1,012 | 1,042 |
| **80%** | 706 | 677 | 0 | 498 | 82 | 776 |
| **90%** | 658 | 587 | 0 | 352 | 24 | 717 |
