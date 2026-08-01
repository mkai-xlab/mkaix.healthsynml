# SE-ResNeXt-50 32x4d: 2026-07-23_06-57-13_multiscale24_native_cam_ce_ema

## Overview

- **Timestamp:** `2026-07-23 06:57:13.378879` UTC
- **Status:** completed
- **Purpose:** This run completed all 30 configured epochs without a runtime error and selected epoch 30 using the EMA validation composite score (0.6728). It tested two changes together: equal fusion of 24x24 and 12x12 class maps, and an exponential moving average with d...
- **Decision / notes:** Rejected; higher map resolution and Grade 1 recall did not offset metric/CAM regressions.

## Configuration

| Item | Value |
| --- | --- |
| Input | 384x384 (resize to 400x400, then crop) |
| ROI and preprocessing | derived from Model Input; exact ROI policy not reported; Square-pad complete ROI; LAB CLAHE; resize 400x400; crop 384x384; ToTensor; ImageNet normalization |
| Loss | Cross-Entropy (CE) |
| Sampler | Full inverse-frequency |
| Pipeline / stages | 3-stage |
| Epochs / selected epoch | 30 (5 warm-up + 15 coarse + 10 fine-tune) / Epoch 30; validation selection score 0.6728 |
| Augmentation | CLAHE(); SquarePad(); transforms.RandomRotation(5); transforms.ColorJitter(brightness=0.08, contrast=0.08); transforms.Resize((400, 400)); transforms.RandomCrop(384); transforms.ToTensor(); transforms.RandomErasing(; transforms.Normalize(; transforms.Center... |
| Heatmap | Equal average of five 24x24 stage-3 class maps and upsampled five 12x12 final-stage class maps; global mean produces logits |

## Recorded Results

| Accuracy | QWK | Macro F1 | Grade 1 Recall | AP | AUC |
| --- | --- | --- | --- | --- | --- |
| 0.5676 | 0.7651 | 0.6220 | 0.5507 | 0.6918 | 0.8703 |

## Files

- Notebook: [`2026-07-23_06-57-13_seresnext50_32x4d_multiscale24_native_cam_ce_ema.ipynb`](2026-07-23_06-57-13_seresnext50_32x4d_multiscale24_native_cam_ce_ema.ipynb), [`legacy_notebook_copy.ipynb`](legacy_notebook_copy.ipynb)
- Figures: `2` file(s) in [`assets/`](assets/)
- Consolidated model report: [SE-ResNeXt-50 32x4d report](../../report.md)
- Structured run index: [experiment_summary.csv](../../experiment_summary.csv)
