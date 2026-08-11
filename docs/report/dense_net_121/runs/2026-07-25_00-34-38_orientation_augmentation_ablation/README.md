# DenseNet-121: 2026-07-25_00-34-38_orientation_augmentation_ablation

## Overview

- **Timestamp:** `2026-07-25 00:34:38`
- **Status:** completed
- **Purpose:** Compare canonicalization with natural-orientation augmentation
- **Decision / notes:** validation winner; later rejected by project requirement

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
| Not recorded | 0.8196 | 0.698 | 0.4575 | 0.7193 | Not recorded |

## Files

- Notebook: [`notebook.ipynb`](../../../../../notebooks/densenet121/experiments/2026-07-25_densenet121_laterality_augmentation_ablation.ipynb)
- Figures: `2` file(s) in [`assets/`](assets/)
- Consolidated model report: [DenseNet-121 report](../../report.md)
- Structured run index: [experiment_summary.csv](../../experiment_summary.csv)
