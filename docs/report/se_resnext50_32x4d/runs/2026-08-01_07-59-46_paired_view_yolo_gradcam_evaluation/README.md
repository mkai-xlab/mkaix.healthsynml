# SE-ResNeXt-50 32x4d: 2026-08-01_07-59-46_paired_view_yolo_gradcam_evaluation

## Overview

- **Timestamp:** `2026-08-01 07:59:46.827053` UTC
- **Status:** completed; not promoted
- **Purpose:** Adapt the selected CE checkpoint to production YOLO square ROIs and evaluate post-hoc final-layer Grad-CAM.
- **Decision / notes:** Model-specific decision recorded in the consolidated report.

## Configuration

| Item | Value |
| --- | --- |
| Input | 384x384 |
| ROI and preprocessing | YOLO box expanded to centered 1.15x square; black padding only outside source image; no center crop; LAB CLAHE 1.25 -> square pad -> resize 384x384 -> ImageNet normalization |
| Loss | Cross-Entropy (CE) |
| Sampler | Full inverse-frequency WeightedRandomSampler with replacement |
| Pipeline / stages | paired-view full-network adaptation |
| Epochs / selected epoch | 5 / 4 |
| Augmentation | 50/50 published/YOLO view; horizontal flip p=0.50; rotation +/-5; brightness/contrast 0.08; RandomErasing p=0.10 scale 0.02-0.05 |
| Heatmap | Post-hoc final-feature Grad-CAM for predicted and true classes |

## Recorded Results

| Accuracy | QWK | Macro F1 | Grade 1 Recall | AP | AUC |
| --- | --- | --- | --- | --- | --- |
| 0.5893719807 | 0.7461351428 | 0.6001785718 | 0.2736486486 | 0.6368493534 | 0.8462351212 |

## Files

- Notebook: [`2026-08-01_07-59-46_seresnext50_32x4d_paired_view_yolo_gradcam_evaluation.ipynb`](2026-08-01_07-59-46_seresnext50_32x4d_paired_view_yolo_gradcam_evaluation.ipynb)
- Figures: `31` file(s) in [`assets/`](assets/)
- Consolidated model report: [SE-ResNeXt-50 32x4d report](../../report.md)
- Structured run index: [experiment_summary.csv](../../experiment_summary.csv)
