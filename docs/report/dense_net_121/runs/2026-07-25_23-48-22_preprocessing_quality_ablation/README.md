# DenseNet-121: 2026-07-25_23-48-22_preprocessing_quality_ablation

## Overview

- **Timestamp:** `2026-07-25 23:48:22`
- **Status:** completed
- **Purpose:** Compare deterministic contrast and padding order
- **Decision / notes:** validation winner

## Configuration

| Item | Value |
| --- | --- |
| Input | 384x384 |
| ROI and preprocessing | clahe1_25_then_pad; Not recorded |
| Loss | cross-entropy |
| Sampler | full inverse-frequency WeightedRandomSampler |
| Pipeline / stages | 5 head warm-up + 15 coarse + 10 full fine-tune |
| Epochs / selected epoch | 5 head warm-up + 15 coarse + 10 full fine-tune / 30 |
| Augmentation | Not recorded |
| Heatmap | final-layer Grad-CAM unless stated otherwise |

## Recorded Results

| Accuracy | QWK | Macro F1 | Grade 1 Recall | AP | AUC |
| --- | --- | --- | --- | --- | --- |
| 0.6695 | 0.8274 | 0.7061 | 0.5294 | 0.7411 | 0.8951 |

## Files

- Notebook: [`notebook.ipynb`](../../../../../notebooks/densenet121/experiments/2026-07-25_densenet121_preprocessing_clahe_ablation.ipynb)
- Figures: `4` file(s) in [`assets/`](assets/)
- Consolidated model report: [DenseNet-121 report](../../report.md)
- Structured run index: [experiment_summary.csv](../../experiment_summary.csv)
