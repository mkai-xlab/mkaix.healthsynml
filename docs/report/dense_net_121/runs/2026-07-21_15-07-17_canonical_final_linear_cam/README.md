# DenseNet-121: 2026-07-21_15-07-17_canonical_final_linear_cam

## Overview

- **Timestamp:** `2026-07-21 15:07:17.633270`
- **Status:** completed
- **Purpose:** This run trained the canonicalfinallinearcam DenseNet-121 architecture for all 30 configured epochs. Right knees were mirrored into the same anatomical orientation as left knees, random horizontal flipping was removed, and a 1x1 convolution produced five sp...
- **Decision / notes:** Not recorded

## Configuration

| Item | Value |
| --- | --- |
| Input | 384x384 (resize to 400x400, then crop) |
| ROI and preprocessing | Square-pad complete ROI; LAB CLAHE; resize 400x400; train random crop or validation/test center crop to 384x384; ToTensor; ImageNet normalization; Not recorded |
| Loss | Cross-Entropy (CE) |
| Sampler | True; full inverse-frequency (samplerpower=1.0) |
| Pipeline / stages | 30 (5 warm-up + 15 coarse + 10 fine-tune) |
| Epochs / selected epoch | 30 (5 warm-up + 15 coarse + 10 fine-tune) / Epoch 23; validation selection score 0.7241 |
| Augmentation | Laterality canonicalization before transforms; horizontal flip disabled; mild rotation/brightness/contrast pipeline; Random Erasing p=0.10; minority augmentation disabled |
| Heatmap | Five bias-free 1x1 class maps; global spatial mean logits; positive predicted-grade native CAM |

## Recorded Results

| Accuracy | QWK | Macro F1 | Grade 1 Recall | AP | AUC |
| --- | --- | --- | --- | --- | --- |
| 0.6534 | 0.8238 | 0.68 | 0.49 | 0.7311 | 0.8978 |

## Files

- Notebook: not stored in this folder
- Figures: `7` file(s) in [`assets/`](assets/)
- Consolidated model report: [DenseNet-121 report](../../report.md)
- Structured run index: [experiment_summary.csv](../../experiment_summary.csv)
