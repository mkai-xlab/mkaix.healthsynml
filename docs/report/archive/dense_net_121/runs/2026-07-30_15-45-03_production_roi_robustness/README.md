# DenseNet-121: 2026-07-30_15-45-03_production_roi_robustness

## Overview

- **Timestamp:** `2026-07-30 15:45:03`
- **Status:** completed
- **Purpose:** Test crop expansion/translation robustness
- **Decision / notes:** rejected: worse aggregate CAM geometry

## Configuration

| Item | Value |
| --- | --- |
| Input | 384x384 |
| ROI and preprocessing | Not recorded; Not recorded |
| Loss | cross-entropy |
| Sampler | full inverse-frequency WeightedRandomSampler |
| Pipeline / stages | 5 head warm-up + 15 coarse + 10 full fine-tune |
| Epochs / selected epoch | 5 head warm-up + 15 coarse + 10 full fine-tune / Not recorded |
| Augmentation | Not recorded |
| Heatmap | final-layer Grad-CAM unless stated otherwise |

## Recorded Results

| Accuracy | QWK | Macro F1 | Grade 1 Recall | AP | AUC |
| --- | --- | --- | --- | --- | --- |
| 0.6186 | 0.765 | 0.6451 | Not recorded | 0.6756 | Not recorded |

## Files

- Notebook: [`notebook.ipynb`](../../../../../../notebooks/densenet121/runs/2026-07-30_15-45-03_production_roi_robustness.ipynb)
- Figures: `25` file(s) in [`assets/`](assets/)
- Consolidated model report: [DenseNet-121 report](../../../../dense_net_121/report.md)
- Structured run index: [experiment_summary.csv](../../../../dense_net_121/experiment_summary.csv)
