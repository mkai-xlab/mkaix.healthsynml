# DenseNet-121: 2026-07-30_09-03-29_paired_view_yolo_roi

## Overview

- **Timestamp:** `2026-07-30 09:03:29`
- **Status:** deployed
- **Purpose:** Adapt base checkpoint to exact production YOLO crop domain
- **Decision / notes:** current production checkpoint

## Configuration

| Item | Value |
| --- | --- |
| Input | 384x384 |
| ROI and preprocessing | CLAHE 1.25 then square pad; Not recorded |
| Loss | cross-entropy |
| Sampler | full inverse-frequency WeightedRandomSampler |
| Pipeline / stages | 5 full-network adaptation epochs |
| Epochs / selected epoch | 5 full-network adaptation epochs / 4 |
| Augmentation | mild training augmentation plus 50/50 published/YOLO view |
| Heatmap | final-layer Grad-CAM unless stated otherwise |

## Recorded Results

| Accuracy | QWK | Macro F1 | Grade 1 Recall | AP | AUC |
| --- | --- | --- | --- | --- | --- |
| 0.597222 | 0.77022 | 0.62152 | 0.375 | 0.669585 | 0.861109 |

## Files

- Notebook: [`evaluation_notebook.ipynb`](../../../../../notebooks/densenet121/runs/2026-07-30_04_evaluate_densenet121_paired_view_yolo_gradcam_384.ipynb), [`train_notebook.ipynb`](../../../../../notebooks/densenet121/runs/2026-07-30_03_train_densenet121_paired_view_yolo_384.ipynb)
- Figures: `11` file(s) in [`assets/`](assets/)
- Consolidated model report: [DenseNet-121 report](../../report.md)
- Structured run index: [experiment_summary.csv](../../experiment_summary.csv)
