# SE-ResNeXt-50 32x4d: 2026-07-23_01-25-36_final_native_cam_ce

## Overview

- **Timestamp:** `2026-07-23 01:25:36.772175` UTC
- **Status:** completed
- **Purpose:** This comparison run completed all 30 configured epochs without a runtime error. The composite validation score selected epoch 24 (0.7003). The selected checkpoint achieved test Accuracy 0.6389, QWK 0.8194, macro F1 0.6671, macro Average Precision 0.7248, an...
- **Decision / notes:** Retained canonical 12x12 native-CAM CE baseline.

## Configuration

| Item | Value |
| --- | --- |
| Input | 384x384 (resize to 400x400, then crop) |
| ROI and preprocessing | derived from Model Input; exact ROI policy not reported; Square-pad complete ROI; LAB CLAHE; resize 400x400; crop 384x384; ToTensor; ImageNet normalization |
| Loss | Cross-Entropy (CE) |
| Sampler | Full inverse-frequency |
| Pipeline / stages | 3-stage |
| Epochs / selected epoch | 30 (5 warm-up + 15 coarse + 10 fine-tune) / Epoch 24; validation selection score 0.7003 |
| Augmentation | CLAHE(); SquarePad(); transforms.RandomRotation(5); transforms.ColorJitter(brightness=0.08, contrast=0.08); transforms.Resize((400, 400)); transforms.RandomCrop(384); transforms.ToTensor(); transforms.RandomErasing(; transforms.Normalize(; transforms.Center... |
| Heatmap | Grad-CAM or CAM method not reported |

## Recorded Results

| Accuracy | QWK | Macro F1 | Grade 1 Recall | AP | AUC |
| --- | --- | --- | --- | --- | --- |
| 0.6389 | 0.8194 | 0.6671 | 0.4155 | 0.7248 | 0.8948 |

## Files

- Notebook: [`2026-07-23_01-25-36_seresnext50_32x4d_final_native_cam_ce.ipynb`](2026-07-23_01-25-36_seresnext50_32x4d_final_native_cam_ce.ipynb)
- Figures: `4` file(s) in [`assets/`](assets/)
- Consolidated model report: [SE-ResNeXt-50 32x4d report](../../report.md)
- Structured run index: [experiment_summary.csv](../../experiment_summary.csv)
