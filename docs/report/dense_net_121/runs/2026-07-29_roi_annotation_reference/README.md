# DenseNet-121: 2026-07-29_roi_annotation_reference

## Overview

- **Timestamp:** `2026-07-29 12:21:26`
- **Status:** completed
- **Purpose:** Reduce published-crop to production-ROI domain shift
- **Decision / notes:** best target-domain arm; superseded by production paired-view run

## Configuration

| Item | Value |
| --- | --- |
| Input | 384x384 |
| ROI and preprocessing | CLAHE 1.25 then square pad; Not recorded |
| Loss | cross-entropy |
| Sampler | full inverse-frequency WeightedRandomSampler |
| Pipeline / stages | 5 head warm-up + 15 coarse + 10 full fine-tune |
| Epochs / selected epoch | 5 head warm-up + 15 coarse + 10 full fine-tune / 5 |
| Augmentation | 50/50 published crop and expanded YOLO ROI |
| Heatmap | final-layer Grad-CAM unless stated otherwise |

## Recorded Results

| Accuracy | QWK | Macro F1 | Grade 1 Recall | AP | AUC |
| --- | --- | --- | --- | --- | --- |
| Not recorded | 0.7189 | 0.6064 | Not recorded | 0.6551 | Not recorded |

## Files

- Notebook: not stored in this folder
- Figures: `10` file(s) in [`assets/`](assets/)
- Consolidated model report: [DenseNet-121 report](../../report.md)
- Structured run index: [experiment_summary.csv](../../experiment_summary.csv)
