# DenseNet-121: 2026-07-30_07-08-32_original_224_ce_3stage

## Overview

- **Timestamp:** `2026-07-30 07:08:32`
- **Status:** completed
- **Purpose:** Train the base model on published 224px crops
- **Decision / notes:** base checkpoint used for paired-view production adaptation

## Configuration

| Item | Value |
| --- | --- |
| Input | 384x384 |
| ROI and preprocessing | CLAHE 1.25 then square pad; Not recorded |
| Loss | cross-entropy |
| Sampler | full inverse-frequency WeightedRandomSampler |
| Pipeline / stages | 5 head warm-up + 15 coarse + 10 full fine-tune |
| Epochs / selected epoch | 5 head warm-up + 15 coarse + 10 full fine-tune / stage 3 |
| Augmentation | Not recorded |
| Heatmap | final-layer Grad-CAM unless stated otherwise |

## Recorded Results

| Accuracy | QWK | Macro F1 | Grade 1 Recall | AP | AUC |
| --- | --- | --- | --- | --- | --- |
| 0.6697 | 0.833 | 0.68 | Not recorded | 0.7305 | 0.898 |

## Files

- Notebook: [`notebook.ipynb`](../../../../../notebooks/densenet121/runs/2026-07-30_07-08-32_original_224_ce_3stage.ipynb)
- Figures: `6` file(s) in [`assets/`](assets/)
- Consolidated model report: [DenseNet-121 report](../../report.md)
- Structured run index: [experiment_summary.csv](../../experiment_summary.csv)
