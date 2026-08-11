# SE-ResNeXt-50 32x4d: 2026-07-25_01-50-53_natural_orientation_flip_gamma_native_cam_ce

## Overview

- **Timestamp:** `2026-07-25 01:50:53.962450` UTC
- **Status:** completed
- **Purpose:** This run completed all 30 configured epochs on a Tesla T4 without a training error and selected epoch 28 using the validation composite score (0.7061). It removed deterministic right-knee canonicalization and instead exposed the model to both orientations w...
- **Decision / notes:** Preferred SE-ResNeXt candidate for single-knee inputs without laterality metadata; external comparison still required.

## Configuration

| Item | Value |
| --- | --- |
| Input | 384x384 (square pad, CLAHE, resize to 400x400, then crop) |
| ROI and preprocessing | derived from Model Input; exact ROI policy not reported; Square-pad complete ROI; LAB CLAHE; resize 400x400; crop 384x384; ToTensor; ImageNet normalization |
| Loss | Cross-Entropy (CE) |
| Sampler | Full inverse-frequency |
| Pipeline / stages | 3-stage |
| Epochs / selected epoch | 30 (5 warm-up + 15 coarse + 10 fine-tune) / Epoch 28; validation selection score 0.7061 |
| Augmentation | Horizontal flip p=0.50; rotation +/-5 degrees; brightness/contrast 0.08; gamma 0.90-1.10 at p=0.20; random erasing p=0.10 |
| Heatmap | Bias-free 1x1 convolution producing five 12x12 grade maps; global spatial mean produces five logits |

## Recorded Results

| Accuracy | QWK | Macro F1 | Grade 1 Recall | AP | AUC |
| --- | --- | --- | --- | --- | --- |
| 0.6558 | 0.8216 | 0.6781 | 0.4122 | 0.7299 | 0.8980 |

## Files

- Notebook: [`2026-07-25_01-50-53_seresnext50_32x4d_natural_orientation_flip_gamma_native_cam_ce.ipynb`](../../../../../../notebooks/seresnext50_32x4d/runs/2026-07-25_se_resnext50_native_cam_orientation_gamma.ipynb)
- Figures: `2` file(s) in [`assets/`](assets/)
- Consolidated model report: [SE-ResNeXt-50 32x4d report](../../../../se_resnext50_32x4d/report.md)
- Structured run index: [experiment_summary.csv](../../../../se_resnext50_32x4d/experiment_summary.csv)
