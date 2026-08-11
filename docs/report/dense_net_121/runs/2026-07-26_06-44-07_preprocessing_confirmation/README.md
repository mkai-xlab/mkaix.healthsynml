# DenseNet-121: 2026-07-26_06-44-07_preprocessing_confirmation

## Overview

- **Timestamp:** `2026-07-26 06:44:07`
- **Status:** completed
- **Purpose:** Uninterrupted two-arm confirmation
- **Decision / notes:** confirmed for subsequent training

## Configuration

| Item | Value |
| --- | --- |
| Input | 384x384 |
| ROI and preprocessing | clahe1_25_then_pad; Not recorded |
| Loss | cross-entropy |
| Sampler | full inverse-frequency WeightedRandomSampler |
| Pipeline / stages | 5 head warm-up + 15 coarse + 10 full fine-tune |
| Epochs / selected epoch | 5 head warm-up + 15 coarse + 10 full fine-tune / Not recorded |
| Augmentation | Not recorded |
| Heatmap | final-layer Grad-CAM unless stated otherwise |

## Recorded Results

| Accuracy | QWK | Macro F1 | Grade 1 Recall | AP | AUC |
| --- | --- | --- | --- | --- | --- |
| Not recorded | 0.8142 | 0.6846 | 0.4837 | 0.7366 | Not recorded |

## Files

- Notebook: [`notebook.ipynb`](../../../../../notebooks/densenet121/experiments/2026-07-26_densenet121_clahe_order_confirmation.ipynb)
- Figures: `2` file(s) in [`assets/`](assets/)
- Consolidated model report: [DenseNet-121 report](../../report.md)
- Structured run index: [experiment_summary.csv](../../experiment_summary.csv)
