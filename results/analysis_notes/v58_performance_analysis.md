# V58 Model Evaluation - Zone 19

This document summarizes the performance findings for the **v58** model (Swin Tiny, Batch Size 256, Learning Rate 5e-5) on the Zone 19 hold-out area.

## Evaluation Summary (1m Buffer)

Evaluated against **1,404** ground truth targets in Zone 19. Extraction performed via Non-Maximum Suppression (NMS) with `min_distance=10px` (1 meter).

| Confidence (%) | Precision | Recall | True Positives (TP) | Total Predictions |
| :--- | :--- | :--- | :--- | :--- |
| **90%** | 0.9127 | 0.2308 | 324 | 352 |
| **80%** | 0.9145 | 0.3276 | 460 | 498 |
| **50%** | 0.8989 | 0.4808 | 675 | 743 |
| **30%** | 0.8493 | 0.5848 | 821 | 958 |
| **20%** | 0.7815 | 0.6481 | 910 | 1,155 |
| **19%** | 0.3649 | 0.6567 | 922 | 2,520 |
| **18%** | 0.2198 | 0.6624 | 930 | 4,228 |
| **17%** | 0.0418 | 0.6681 | 938 | 22,485 |
| **10%** | 0.0403 | 0.7130 | 1,001 | 24,872 |
| **5%** | 0.0394 | 0.7721 | 1,084 | 27,584 |

## Comparison with 3m Buffer (High Confidence)

| Confidence (%) | Precision (1m) | Recall (1m) | Precision (3m) | Recall (3m) |
| :--- | :--- | :--- | :--- | :--- |
| **90%** | 0.9127 | 0.2308 | 0.9648 | 0.3305 |
| **80%** | 0.9145 | 0.3276 | 0.9635 | 0.4509 |
| **50%** | 0.8989 | 0.4808 | 0.9566 | 0.6218 |

## Operational Recommendations

1.  **Recommended Threshold**: Use **20% confidence** for high-recall operational needs. This significantly boosts recall (64.8%) while maintaining a usable precision (78.2%).
2.  **The "Noise Floor"**: Avoid thresholds below **18%**. There is a performance "cliff" between 18% and 17% where the number of candidates jumps by **5.3x** for only a **0.6%** recall gain.
3.  **Buffer Choice**: In magnetic target detection, a **3m buffer** is typically more representative of field success due to the displacement of peak anomalies from physical centroids.
