# DenseNet-121: 2026-07-28_04-58-51_roi_robustness_ablation

## Overview

- **Timestamp:** `2026-07-28 04:58:51`
- **Status:** completed
- **Purpose:** Test geometry/acquisition robustness without changing loss
- **Decision / notes:** rejected

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
| Not recorded | 0.8 | 0.6625 | 0.451 | 0.7329 | Not recorded |

## Files

- Notebook: [`notebook.ipynb`](../../../../../notebooks/densenet121/experiments/2026-07-28_densenet121_roi_robustness_ablation.ipynb)
- Figures: `2` file(s) in [`assets/`](assets/)
- Consolidated model report: [DenseNet-121 report](../../report.md)
- Structured run index: [experiment_summary.csv](../../experiment_summary.csv)
