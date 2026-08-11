# DenseNet-121: 2026-07-20_12-36-36_corn

## Overview

- **Timestamp:** `2026-07-20 12:36:36`
- **Status:** completed
- **Purpose:** This run trained DenseNet-121 through all three stages for 45 epochs using Conditional Ordinal (CORN) loss. Images were resized to 400x400 and cropped to 384x384; validation and test inference used a single center crop with no TTA. Training used the class-b...
- **Decision / notes:** Not recorded

## Configuration

| Item | Value |
| --- | --- |
| Input | 384x384 (resize to 400x400, then crop) |
| ROI and preprocessing | Square-pad complete ROI; LAB CLAHE; resize 400x400; crop 384x384; ToTensor; ImageNet normalization; Not recorded |
| Loss | corn |
| Sampler | True |
| Pipeline / stages | 45 (5 warm-up + 25 coarse + 15 fine-tune) |
| Epochs / selected epoch | 45 (5 warm-up + 25 coarse + 15 fine-tune) / Not recorded |
| Augmentation | Horizontal flip p=0.50; rotation +/-8 degrees; random crop 384x384; one Random Erasing p=0.10, scale 0.02-0.05, ratio 0.5-2.0; minority augmentation disabled |
| Heatmap | Standard Grad-CAM over three normalized scales |

## Recorded Results

| Accuracy | QWK | Macro F1 | Grade 1 Recall | AP | AUC |
| --- | --- | --- | --- | --- | --- |
| 0.6715 | 0.8246 | 0.68 | 0.31 | 0.7337 | 0.8963 |

## Files

- Notebook: [`notebook.ipynb`](../../../../../../notebooks/densenet121/runs/2026-07-20_12-36-36_corn.ipynb)
- Figures: `6` file(s) in [`assets/`](assets/)
- Consolidated model report: [DenseNet-121 report](../../../../dense_net_121/report.md)
- Structured run index: [experiment_summary.csv](../../../../dense_net_121/experiment_summary.csv)
