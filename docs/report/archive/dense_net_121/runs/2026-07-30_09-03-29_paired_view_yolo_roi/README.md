# DenseNet-121: 2026-07-30_09-03-29_paired_view_yolo_roi

## Overview

- **Timestamp:** `2026-07-30 09:03:29`
- **Status:** deployed
- **Purpose:** Adapt base checkpoint to exact production YOLO crop domain
- **Decision / notes:** current production checkpoint

## Configuration

| Item | Value |
| --- | --- |
| Input | 384x384 |
| ROI and preprocessing | CLAHE 1.25 then square pad; Not recorded |
| Loss | cross-entropy |
| Sampler | full inverse-frequency WeightedRandomSampler |
| Pipeline / stages | 5 full-network adaptation epochs |
| Epochs / selected epoch | 5 full-network adaptation epochs / 4 |
| Augmentation | mild training augmentation plus 50/50 published/YOLO view |
| Heatmap | final-layer Grad-CAM unless stated otherwise |

## Recorded Results

| Accuracy | QWK | Macro F1 | Grade 1 Recall | AP | AUC |
| --- | --- | --- | --- | --- | --- |
| 0.597222 | 0.77022 | 0.62152 | 0.375 | 0.669585 | 0.861109 |

## Files

- Notebook sources: not retained; this archived folder preserves the recorded 384x384 metrics and Grad-CAM figures.
- Figures: `11` file(s) in [`assets/`](assets/)
- Consolidated model report: [DenseNet-121 report](../../../../dense_net_121/report.md)
- Structured run index: [experiment_summary.csv](../../../../dense_net_121/experiment_summary.csv)
