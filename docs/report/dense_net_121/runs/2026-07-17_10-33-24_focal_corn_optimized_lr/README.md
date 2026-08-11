# DenseNet-121: 2026-07-17_10-33-24_focal_corn_optimized_lr

## Overview

- **Timestamp:** `2026-07-17 10:33:24`
- **Status:** completed
- **Purpose:** This run successfully trained a densenet121 model in standard 1-stage mode for 45 epochs on 224x224 images using Focal CORN loss. By enabling the class-balancing WeightedRandomSampler, minority augmentations, and double Cutout (Random Erasing), the model ac...
- **Decision / notes:** Not recorded

## Configuration

| Item | Value |
| --- | --- |
| Input | 224x224 |
| ROI and preprocessing | Square-pad complete ROI; LAB CLAHE (OpenCV implementation); resize to 224x224; ToTensor; ImageNet mean/std normalization; Not recorded |
| Loss | focalcorn |
| Sampler | True |
| Pipeline / stages | 30 (Actual: 45) |
| Epochs / selected epoch | 30 (Actual: 45) / Not recorded |
| Augmentation | Horizontal flip p=0.50; rotation +/-8 degrees; double Random Erasing p=0.80 for regular train and p=0.90 for minority transform, scale 0.02-0.15, ratio 0.3-3.3; minority augmentation enabled |
| Heatmap | Grad-CAM or CAM method not reported |

## Recorded Results

| Accuracy | QWK | Macro F1 | Grade 1 Recall | AP | AUC |
| --- | --- | --- | --- | --- | --- |
| 0.6612 | 0.8271 | 0.67 | 0.41 | 0.7280 | 0.8984 |

## Files

- Notebook: [`notebook.ipynb`](../../../../../notebooks/densenet121/runs/2026-07-17_10-33-24_focal_corn_optimized_lr.ipynb)
- Figures: `6` file(s) in [`assets/`](assets/)
- Consolidated model report: [DenseNet-121 report](../../report.md)
- Structured run index: [experiment_summary.csv](../../experiment_summary.csv)
