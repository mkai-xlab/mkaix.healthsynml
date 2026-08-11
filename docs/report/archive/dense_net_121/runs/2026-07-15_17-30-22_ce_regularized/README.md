# DenseNet-121: 2026-07-15_17-30-22_ce_regularized

## Overview

- **Timestamp:** `2026-07-15 17:30:22`
- **Status:** completed
- **Purpose:** This run successfully trained a densenet121 model in standard 1-stage mode for 19 epochs on 224x224 images using Cross-Entropy (CE) loss. By enabling the class-balancing WeightedRandomSampler, minority augmentations, and double Cutout (Random Erasing), the...
- **Decision / notes:** Not recorded

## Configuration

| Item | Value |
| --- | --- |
| Input | 224x224 |
| ROI and preprocessing | Square-pad complete ROI; LAB CLAHE (OpenCV implementation); resize to 224x224; ToTensor; ImageNet mean/std normalization; Not recorded |
| Loss | ce |
| Sampler | True |
| Pipeline / stages | 30 (Actual: 19) |
| Epochs / selected epoch | 30 (Actual: 19) / Not recorded |
| Augmentation | Horizontal flip p=0.50; rotation +/-8 degrees; double Random Erasing p=0.80 for regular train and p=0.90 for minority transform, scale 0.02-0.15, ratio 0.3-3.3; minority augmentation enabled |
| Heatmap | Grad-CAM or CAM method not reported |

## Recorded Results

| Accuracy | QWK | Macro F1 | Grade 1 Recall | AP | AUC |
| --- | --- | --- | --- | --- | --- |
| 0.6594 | 0.8283 | 0.68 | 0.49 | 0.7287 | 0.8993 |

## Files

- Notebook: [`notebook.ipynb`](../../../../../../notebooks/densenet121/runs/2026-07-15_17-30-22_ce_regularized.ipynb)
- Figures: `6` file(s) in [`assets/`](assets/)
- Consolidated model report: [DenseNet-121 report](../../../../dense_net_121/report.md)
- Structured run index: [experiment_summary.csv](../../../../dense_net_121/experiment_summary.csv)
