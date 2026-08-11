# DenseNet-121: 2026-07-25_06-30-25_final_noncanonical_loss_ablation

## Overview

- **Timestamp:** `2026-07-25 06:30:25`
- **Status:** completed
- **Purpose:** Compare CE and ordinal objectives under one split/config
- **Decision / notes:** selected; locked test QWK 0.8197

## Configuration

| Item | Value |
| --- | --- |
| Input | 384x384 |
| ROI and preprocessing | Not recorded; Not recorded |
| Loss | CE |
| Sampler | full inverse-frequency WeightedRandomSampler |
| Pipeline / stages | 5 head warm-up + 15 coarse + 10 full fine-tune |
| Epochs / selected epoch | 5 head warm-up + 15 coarse + 10 full fine-tune / Not recorded |
| Augmentation | Not recorded |
| Heatmap | final-layer Grad-CAM unless stated otherwise |

## Recorded Results

| Accuracy | QWK | Macro F1 | Grade 1 Recall | AP | AUC |
| --- | --- | --- | --- | --- | --- |
| 0.6465 | 0.8083 | 0.6819 | 0.3856 | 0.731 | Not recorded |

## Files

- Notebook: [`executed_notebook.ipynb`](../../../../../../notebooks/densenet121/archive/2026-07-25_densenet201_noncanonical_loss_ablation_executed.ipynb), [`notebook.ipynb`](../../../../../../notebooks/densenet121/experiments/2026-07-25_densenet121_natural_orientation_loss_ablation.ipynb)
- Figures: `2` file(s) in [`assets/`](assets/)
- Consolidated model report: [DenseNet-121 report](../../../../dense_net_121/report.md)
- Structured run index: [experiment_summary.csv](../../../../dense_net_121/experiment_summary.csv)
