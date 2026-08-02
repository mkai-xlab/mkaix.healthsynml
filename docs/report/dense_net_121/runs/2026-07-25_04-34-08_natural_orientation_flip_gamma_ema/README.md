# DenseNet-121: 2026-07-25_04-34-08_natural_orientation_flip_gamma_ema

## Overview

- **Timestamp:** `2026-07-25 04:34:08.758611`
- **Status:** rejected
- **Purpose:** This run completed all 30 configured epochs and evaluated the validation-selected epoch-30 EMA checkpoint on the test set. It tested natural left/right orientation instead of deterministic right-knee mirroring, horizontal flipping during training, mild rand...
- **Decision / notes:** Model-specific decision recorded in the consolidated report.

## Configuration

| Item | Value |
| --- | --- |
| Input | 384x384 (resize to 400x400, then crop) |
| ROI and preprocessing | CLAHE(clipLimit=self.cliplimit, tileGridSize=self.tilegridsize); OpenCVCLAHE(); transforms.RandomHorizontalFlip(p=TrainingConfig.horizontalflipp); transforms.RandomRotation(degrees=TrainingConfig.rotationdegrees); transforms.ColorJitter(; transforms.Resize(...; Not recorded |
| Loss | Cross-Entropy (CE) in all stages |
| Sampler | True; full inverse-frequency (samplerpower=1.0) in all stages |
| Pipeline / stages | Not recorded |
| Epochs / selected epoch | Not recorded / 30 / 30; validation selection score 0.6693 |
| Augmentation | Random Horizontal Flip: p=0.50, training only; Minority Augmentations / TTA: False / False; Gamma / Brightness / Contrast: Gamma 0.90-1.10 at p=0.20; brightness and contrast jitter 0.08; Rotation / Random Erasing: +/-5 degrees; erasing p=0.10 |
| Heatmap | Grad-CAM or CAM method not reported |

## Recorded Results

| Accuracy | QWK | Macro F1 | Grade 1 Recall | AP | AUC |
| --- | --- | --- | --- | --- | --- |
| 0.4553 | 0.7248 | 0.5334 | Not recorded | 0.6663 | 0.8619 |

## Files

- Notebook: [`notebook.ipynb`](notebook.ipynb)
- Figures: `2` file(s) in [`assets/`](assets/)
- Consolidated model report: [DenseNet-121 report](../../report.md)
- Structured run index: [experiment_summary.csv](../../experiment_summary.csv)
