# DenseNet-121: 2026-07-31_00-49-00_glcm_fusion_comparison

## Overview

- **Timestamp:** `2026-07-31 00:49:00`
- **Status:** completed
- **Purpose:** Test whether texture features improve the production checkpoint
- **Decision / notes:** Model-specific decision recorded in the consolidated report.

## Configuration

| Item | Value |
| --- | --- |
| Input | 384x384 |
| ROI and preprocessing | Not recorded; Not recorded |
| Loss | cross-entropy |
| Sampler | full inverse-frequency WeightedRandomSampler |
| Pipeline / stages | 5 head warm-up + 15 coarse + 10 full fine-tune |
| Epochs / selected epoch | 5 head warm-up + 15 coarse + 10 full fine-tune / 5 |
| Augmentation | Not recorded |
| Heatmap | final-layer Grad-CAM unless stated otherwise |

## Recorded Results

| Accuracy | QWK | Macro F1 | Grade 1 Recall | AP | AUC |
| --- | --- | --- | --- | --- | --- |
| 0.6235 | 0.77368 | 0.6519 | 0.4379 | 0.69634 | 0.87394 |

## Files

- Notebook: [`notebook.ipynb`](notebook.ipynb)
- Figures: none stored in this folder
- Consolidated model report: [DenseNet-121 report](../../report.md)
- Structured run index: [experiment_summary.csv](../../experiment_summary.csv)
