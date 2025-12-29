# V60 Model Evaluation - Zone 19 (CNN Stability Refinement)

This document summarizes the performance findings for the **v60** model (UNet CNN with Gradient Accumulation, Softened Deep Supervision) on the Zone 19 hold-out area.

## Technical Changes vs V59

To address the training instability and low recall observed in the v59 baseline, several stability refinements were implemented:

1.  **Gradient Accumulation**: Introduced `accumulation_steps=4`. With a base batch size of 128, the **effective batch size is now 512**. This stabilizes gradient updates for sparse targets.
2.  **Softened Deep Supervision**: Multi-scale loss weighting was rebalanced to favor the final high-resolution output:
    *   **V59**: `(0.5 * loss4) + (0.3 * loss3) + (0.2 * loss2)`
    *   **V60**: `(1.0 * loss4) + (0.1 * loss3) + (0.05 * loss2)`
3.  **Reduced Noise Augmentation**: Decreased Gaussian noise factor from **0.05** to **0.01** to prevent overwriting weak magnetic signals during training.

## Evaluation Summary (1m Buffer)

Evaluated against **1,404** ground truth targets in Zone 19. Extraction performed via Non-Maximum Suppression (NMS) with `min_distance=10px` (1 meter).

| Confidence (%) | Precision | Recall | True Positives (TP) | Total Predictions |
| :--- | :--- | :--- | :--- | :--- |
| **90%** | 0.6538 | 0.0121 | 17 | 24 |
| **80%** | 0.7412 | 0.0449 | 63 | 82 |
| **50%** | 0.8629 | 0.6275 | 881 | 1012 |
| **30%** | 0.7539 | 0.7493 | 1052 | 1388 |
| **20%** | 0.0635 | 0.8048 | 1130 | 17821 |
| **10%** | 0.0557 | 0.8611 | 1209 | 21768 |

## Comparison with 3m Buffer (High Confidence)

| Confidence (%) | Precision (1m) | Recall (1m) | Precision (3m) | Recall (3m) |
| :--- | :--- | :--- | :--- | :--- |
| **90%** | 0.6538 | 0.0121 | 0.8485 | 0.0185 |
| **80%** | 0.7412 | 0.0449 | 0.9355 | 0.0783 |
| **50%** | 0.8629 | 0.6275 | 0.9432 | 0.7664 |

## Comparison vs V59 (CNN Baseline)

| Metric (@50% Conf, 1m) | V59 (Baseline) | V60 (Stability) | Improvement |
| :--- | :--- | :--- | :--- |
| **Precision** | 0.9070 | 0.8629 | -4.8% |
| **Recall** | 0.3604 | 0.6275 | **+74.1%** |
| **TP** | 506 | 881 | **+375 targets** |

## Operational Recommendations

1.  **Massive Recall Gain**: V60 shows a transformative improvement in recall (+74%) compared to V59. This suggests that the stability refinements (Gradient Accumulation and Softened Deep Supervision) allowed the model to learn much more effectively.
2.  **Recommended Threshold**: Use **50% confidence** for a great balance of Precision (86%) and Recall (62.7% @ 1m, 76.6% @ 3m).
3.  **Noise Floor**: The "noise cliff" is even more pronounced in V60, starting around 30% confidence. Below 30%, the false positive count explodes (from 1,388 to 17,821).
4.  **Conclusion**: V60 is a significantly better CNN baseline than V59. It approaches V58 (SwinUNet) levels of recall while maintaining solid CNN-style precision at high thresholds.
