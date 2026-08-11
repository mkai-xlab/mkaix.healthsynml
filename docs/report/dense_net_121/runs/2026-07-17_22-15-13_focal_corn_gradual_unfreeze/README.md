# DenseNet-121: 2026-07-17_22-15-13_focal_corn_gradual_unfreeze

## Overview

- **Timestamp:** `2026-07-17 22:15:13`
- **Status:** invalid implementation; historical result only
- **Purpose:** This run successfully trained a densenet121 model in standard 1-stage mode for 45 epochs on 224x224 images using Focal CORN loss. By enabling the class-balancing WeightedRandomSampler, minority augmentations, and double Cutout (Random Erasing), the model ac...
- **Decision / notes:** Do not use for model selection or production.

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
| 0.6498 | 0.7564 | 0.59 | 0.00 | 0.7059 | 0.8814 |

## Files

- Notebook: `notebook.ipynb` not retained (invalid implementation)
- Figures: `6` file(s) in [`assets/`](assets/)
- Consolidated model report: [DenseNet-121 report](../../report.md)
- Structured run index: [experiment_summary.csv](../../experiment_summary.csv)
