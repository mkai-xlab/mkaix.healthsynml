# DenseNet-121: 2026-07-15_13-42-33_ce_baseline

## Overview

- **Timestamp:** `2026-07-15 13:42:33`
- **Status:** completed
- **Purpose:** This run successfully trained a densenet121 model in standard 1-stage mode for 30 epochs on 224x224 images using Cross-Entropy (CE) loss. By enabling the class-balancing WeightedRandomSampler, minority augmentations, and double Cutout (Random Erasing), the...
- **Decision / notes:** Not recorded

## Configuration

| Item | Value |
| --- | --- |
| Input | 224x224 |
| ROI and preprocessing | Square-pad complete ROI; LAB CLAHE (OpenCV implementation); resize to 224x224; ToTensor; ImageNet mean/std normalization; Not recorded |
| Loss | ce |
| Sampler | False |
| Pipeline / stages | 30 (Actual: 30) |
| Epochs / selected epoch | 30 (Actual: 30) / Not recorded |
| Augmentation | Horizontal flip p=0.50; rotation +/-8 degrees; Random Erasing disabled; no minority augmentation |
| Heatmap | Grad-CAM or CAM method not reported |

## Recorded Results

| Accuracy | QWK | Macro F1 | Grade 1 Recall | AP | AUC |
| --- | --- | --- | --- | --- | --- |
| 0.6691 | 0.8058 | 0.67 | 0.22 | 0.7009 | 0.8798 |

## Files

- Notebook: [`notebook.ipynb`](notebook.ipynb)
- Figures: `7` file(s) in [`assets/`](assets/)
- Consolidated model report: [DenseNet-121 report](../../report.md)
- Structured run index: [experiment_summary.csv](../../experiment_summary.csv)
