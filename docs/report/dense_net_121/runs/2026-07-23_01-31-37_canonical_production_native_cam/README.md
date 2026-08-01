# DenseNet-121: 2026-07-23_01-31-37_canonical_production_native_cam

## Overview

- **Timestamp:** `2026-07-23 01:31:37.184239`
- **Status:** historical production
- **Purpose:** This run completed 27 of the 30 configured epochs and selected epoch 27 using the validation composite score (0.7276). The exact saved checkpoint was copied to checkpoints/densenet121/bestmodel.pth. On the test set it achieved Accuracy 0.6612, QWK 0.8178, m...
- **Decision / notes:** Not recorded

## Configuration

| Item | Value |
| --- | --- |
| Input | 384x384 (resize to 400x400, then crop) |
| ROI and preprocessing | Square-pad complete ROI; LAB CLAHE; resize 400x400; train random crop or validation/test center crop to 384x384; ToTensor; ImageNet normalization; Not recorded |
| Loss | Cross-Entropy (CE) |
| Sampler | True; full inverse-frequency (samplerpower=1.0) |
| Pipeline / stages | 30 (5 warm-up + 15 coarse + 10 fine-tune) |
| Epochs / selected epoch | 30 (5 warm-up + 15 coarse + 10 fine-tune) / 27 / 27; validation selection score 0.7276 |
| Augmentation | Laterality canonicalization before transforms; horizontal flip disabled; mild rotation/brightness/contrast pipeline; Random Erasing p=0.10; minority augmentation disabled |
| Heatmap | Five bias-free 1x1 class maps; global spatial mean logits; positive predicted-grade native CAM |

## Recorded Results

| Accuracy | QWK | Macro F1 | Grade 1 Recall | AP | AUC |
| --- | --- | --- | --- | --- | --- |
| 0.6612 | 0.8178 | 0.6811 | 0.4493 | 0.7334 | 0.8987 |

## Files

- Notebook: not stored in this folder
- Figures: `2` file(s) in [`assets/`](assets/)
- Consolidated model report: [DenseNet-121 report](../../report.md)
- Structured run index: [experiment_summary.csv](../../experiment_summary.csv)
