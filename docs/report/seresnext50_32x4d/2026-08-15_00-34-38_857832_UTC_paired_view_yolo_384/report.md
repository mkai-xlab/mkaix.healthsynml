# SE-ResNeXt-50 Paired-View YOLO ROI Evaluation

| Field | Value |
| --- | --- |
| Evaluation timestamp | 2026-08-15 00:34:38 UTC |
| Model | SE-ResNeXt-50 32x4d |
| Checkpoint | `/content/drive/MyDrive/Models/seresnext50_32x4d_yolo_384/2026-08-14_23-59-46_940080_UTC/best_model.pth` |
| Architecture | `seresnext50_32x4d_linear_gradcam` |
| Input | 384x384 YOLO ROI |
| Preprocessing | CLAHE clip limit 1.25, square padding, resize to 384, ImageNet normalization |
| Evaluation split | Locked YOLO ROI test split |
| Test samples | 1,656 |
| Heatmap | Post-hoc Grad-CAM from final SE-ResNeXt convolutional layer |

## Metrics

| Metric | Result |
| --- | ---: |
| Accuracy | 0.5876 |
| QWK | 0.7437 |
| Macro precision | 0.6088 |
| Macro recall | 0.6155 |
| Macro F1 | 0.6079 |
| Macro AP | 0.6377 |
| Macro ROC-AUC | 0.8507 |

## Per-Grade Results

| Grade | Precision | Recall | F1 | Support |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.6900 | 0.7418 | 0.7149 | 639 |
| 1 | 0.2629 | 0.3277 | 0.2917 | 296 |
| 2 | 0.6111 | 0.4430 | 0.5136 | 447 |
| 3 | 0.7385 | 0.7220 | 0.7302 | 223 |
| 4 | 0.7414 | 0.8431 | 0.7890 | 51 |

## Confusion Matrix

Rows are true grades and columns are predicted grades:

```text
[[474, 118,  40,   6,  1],
 [134,  97,  59,   5,  1],
 [ 76, 133, 198,  40,  0],
 [  3,  21,  25, 161, 13],
 [  0,   0,   2,   6, 43]]
```

There were 683 misclassified cases. The main errors are adjacent-grade confusions, especially Grade 1 with Grades 0/2 and Grade 2 with Grades 1/3.

## Artifacts

- Executed notebook: `04_evaluate_se_resnext50_yolo_roi_384_gradcam.ipynb`
- Confusion matrix: generated at `evaluation_2026-08-15_00-34-38_857832_UTC/confusion_matrix.png` in the checkpoint run directory
- Correct Grad-CAM grids: `evaluation_2026-08-15_00-34-38_857832_UTC/gradcam_correct_by_true_grade`
- Misclassified true/predicted Grad-CAM pairs: `evaluation_2026-08-15_00-34-38_857832_UTC/gradcam_misclassified_pairs`

This report records the completed evaluation only; it does not tune the test set or claim that Grad-CAM is lesion segmentation.
