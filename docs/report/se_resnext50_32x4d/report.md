# SE-ResNeXt-50 Training Execution Log
This file records the optimized SE-ResNeXt-50 comparison run, its exact configuration, predictive metrics, and native-CAM audit.

## Model Performance Summary

| Run Timestamp | Model Configuration / Loss | Accuracy | QWK Score | ROC AUC | Avg Precision | Macro F1 | Grade 1 Recall | CAM Joint Energy | CAM Border Energy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-07-23 01:25:36.772175 UTC | **Final Linear Native CAM (Laterality Canonicalized)**<br>Cross-Entropy (CE) | 0.6389 (95% bootstrap CI: 0.6153 - 0.6624) | 0.8194 (95% bootstrap CI: 0.7999 - 0.8384) | 0.8948 | 0.7248 | 0.6671 | 0.4155 | 0.8707 | 0.0749 |

## Run: 2026-07-23 01:25:36.772175 UTC (SE-RESNEXT50-32X4D - Final Linear Native CAM)

### Summary
This comparison run completed all 30 configured epochs without a runtime error. The composite validation score selected epoch 24 (`0.7003`). The selected checkpoint achieved test Accuracy `0.6389`, QWK `0.8194`, macro F1 `0.6671`, macro Average Precision `0.7248`, and macro ROC AUC `0.8948`. This is the optimized SE-ResNeXt configuration selected by the preceding controlled CAM ablation; it is not the application checkpoint.

### Configurations

| Parameter | Value |
| --- | --- |
| **Model** | `seresnext50_32x4d` |
| **Architecture** | `final_native_cam_ce` |
| **Model Input** | 384x384 (resize to 400x400, then crop) |
| **Pipeline** | 3-stage |
| **Epochs** | 30 (5 warm-up + 15 coarse + 10 fine-tune) |
| **Selected Checkpoint** | Epoch 24; validation selection score 0.7003 |
| **Loss Function** | Cross-Entropy (CE) |
| **Balanced Sampler** | Full inverse-frequency |
| **Laterality Canonicalization** | True; right knees mirrored before transforms |
| **Batch Size / Workers / GPU** | 48 / 4 / Tesla T4 |
| **Warm-up Learning Rate** | 3e-4; native-CAM head only |
| **Coarse Learning Rates** | Backbone 3e-5; native-CAM head 3e-4 |
| **Fine-tune Learning Rate** | 1e-5; full model |
| **Weight Decay** | 1e-4 in warm-up/coarse; 1e-3 in fine-tuning |
| **Checkpoint Directory** | `2026-07-23_01-25-36_772175_UTC_final_native_cam_ce` |
| **Executed Notebook Archive** | [`2026-07-23_01-25-36_seresnext50_32x4d_final_native_cam_ce.ipynb`](2026-07-23_01-25-36_seresnext50_32x4d_final_native_cam_ce.ipynb) |

### Selected Validation Metrics

| Metric | Score |
| --- | --- |
| **QWK Score** | 0.7873 |
| **Macro F1** | 0.6498 |
| **Grade 1 Recall** | 0.4183 |
| **Average Precision** | 0.6915 |
| **Composite Selection Score** | 0.7003 |

The completed notebook did not print the selected epoch's validation Accuracy, macro Recall, or AUC. They are therefore not reconstructed or invented here.

### Final Test Metrics

| Metric | Score |
| --- | --- |
| **Accuracy** | 0.6389 (95% bootstrap CI: 0.6153 - 0.6624) |
| **QWK Score** | 0.8194 (95% bootstrap CI: 0.7999 - 0.8384) |
| **Macro Precision** | 0.6677 |
| **Macro Recall** | 0.6727 |
| **Macro F1** | 0.6671 |
| **Grade 1 Recall** | 0.4155 |
| **Average Precision** | 0.7248 |
| **ROC AUC** | 0.8948 |
| **Loss** | 0.7905 |

The Accuracy and QWK intervals were reconstructed with `5,000` multinomial resamples of the saved test confusion matrix. They are image-level intervals because patient identifiers were not exported with the notebook result.

### Classification Report

```text
              precision    recall  f1-score   support

     Grade 0       0.77      0.68      0.72       639
     Grade 1       0.31      0.42      0.36       296
     Grade 2       0.69      0.60      0.64       447
     Grade 3       0.75      0.84      0.79       223
     Grade 4       0.82      0.82      0.82        51

    accuracy                           0.64      1656
   macro avg       0.67      0.67      0.67      1656
weighted avg       0.66      0.64      0.65      1656
```

The test confusion matrix was:

```text
             Pred 0  Pred 1  Pred 2  Pred 3  Pred 4
True Grade 0    437     170      29       3       0
True Grade 1     99     123      71       3       0
True Grade 2     34      96     269      48       0
True Grade 3      0       5      22     187       9
True Grade 4      0       0       0       9      42
```

### Epoch-by-Epoch Training History

| Stage | Epoch | QWK | Macro F1 | Grade 1 Recall | AP | Selection |
| --- | --- | --- | --- | --- | --- | --- |
| Warm-up | 1 | 0.3197 | 0.2757 | 0.1307 | 0.3406 | 0.3160 |
| Warm-up | 2 | 0.4070 | 0.2244 | 0.0327 | 0.3540 | 0.3368 |
| Warm-up | 3 | 0.4508 | 0.2715 | 0.5294 | 0.3807 | 0.4187 |
| Warm-up | 4 | 0.4172 | 0.2437 | 0.1373 | 0.3824 | 0.3610 |
| Warm-up | 5 | 0.4825 | 0.3133 | 0.3399 | 0.3919 | 0.4244 |
| Coarse | 6 | 0.6480 | 0.4476 | 0.0261 | 0.5210 | 0.5236 |
| Coarse | 7 | 0.6946 | 0.5174 | 0.0654 | 0.6004 | 0.5781 |
| Coarse | 8 | 0.7600 | 0.6060 | 0.1569 | 0.6466 | 0.6429 |
| Coarse | 9 | 0.7537 | 0.5740 | 0.2092 | 0.6494 | 0.6356 |
| Coarse | 10 | 0.7012 | 0.5898 | 0.3464 | 0.6496 | 0.6369 |
| Coarse | 11 | 0.7603 | 0.6113 | 0.3333 | 0.6679 | 0.6670 |
| Coarse | 12 | 0.7777 | 0.6306 | 0.3725 | 0.6740 | 0.6825 |
| Coarse | 13 | 0.7574 | 0.6215 | 0.3660 | 0.6775 | 0.6740 |
| Coarse | 14 | 0.7655 | 0.6301 | 0.4118 | 0.6801 | 0.6836 |
| Coarse | 15 | 0.7747 | 0.6380 | 0.3333 | 0.6839 | 0.6818 |
| Coarse | 16 | 0.7839 | 0.6361 | 0.3464 | 0.6889 | 0.6870 |
| Coarse | 17 | 0.7800 | 0.6352 | 0.3529 | 0.6882 | 0.6857 |
| Coarse | 18 | 0.7800 | 0.6415 | 0.3595 | 0.6870 | 0.6879 |
| Coarse | 19 | 0.7795 | 0.6393 | 0.3464 | 0.6898 | 0.6862 |
| Coarse | 20 | 0.7791 | 0.6358 | 0.3333 | 0.6884 | 0.6833 |
| Fine-tune | 21 | 0.7745 | 0.6364 | 0.3399 | 0.6900 | 0.6824 |
| Fine-tune | 22 | 0.7746 | 0.6379 | 0.3529 | 0.6902 | 0.6851 |
| Fine-tune | 23 | 0.7711 | 0.6329 | 0.3595 | 0.6910 | 0.6823 |
| Fine-tune | 24 | **0.7873** | **0.6498** | **0.4183** | 0.6915 | **0.7003** |
| Fine-tune | 25 | 0.7860 | 0.6452 | 0.3660 | 0.6920 | 0.6929 |
| Fine-tune | 26 | 0.7811 | 0.6488 | 0.4183 | 0.6982 | 0.6983 |
| Fine-tune | 27 | 0.7814 | 0.6472 | 0.4118 | **0.7012** | 0.6979 |
| Fine-tune | 28 | 0.7850 | 0.6530 | 0.4052 | 0.7002 | 0.7001 |
| Fine-tune | 29 | 0.7883 | 0.6554 | 0.3856 | 0.6998 | 0.7000 |
| Fine-tune | 30 | 0.7837 | 0.6568 | 0.3987 | 0.7011 | 0.7000 |

### Visualizations

#### Confusion Matrix, ROC, and Precision-Recall Curves

![SE-ResNeXt test metrics, run 2026-07-23 01:25:36.772175 UTC](assets/2026-07-23_01-25-36_test_metrics.png)

#### Native-CAM Audit and Worst Cases

![SE-ResNeXt native-CAM audit, run 2026-07-23 01:25:36.772175 UTC](assets/2026-07-23_01-25-36_native_cam_audit.png)

### Native-CAM Evaluation

The audit used `227` validation cases, with up to 50 cases per grade. Mean predicted-map joint energy was `0.8707`, border energy was `0.0749`, lower-tibia energy was `0.0880`, and the peak-inside-joint rate was `0.9956`. These broad-region results are better than the completed DenseNet run (`0.7996`, `0.1323`, `0.1006`, and `1.0000`, respectively), except for the negligible peak-rate difference.

The visual audit still contains lateral marginal hotspots. A rectangular joint band can count these as correct, so the reported energy values do not establish exact localization of joint-space narrowing or osteophytes. The native map is faithful to the class logit because global averaging of that map produces the logit, but it remains a coarse `12x12` explanation rather than an expert lesion mask.

### Comparison and Decision

| Model | Accuracy | QWK | Macro F1 | Grade 1 Recall | AP | AUC | Joint Energy | Border Energy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **DenseNet-121** | **0.6612** | 0.8178 | **0.6811** | **0.4493** | **0.7334** | **0.8987** | 0.7996 | 0.1323 |
| SE-ResNeXt-50 | 0.6389 | **0.8194** | 0.6671 | 0.4155 | 0.7248 | 0.8948 | **0.8707** | **0.0749** |

SE-ResNeXt provides better broad-ROI CAM concentration and a QWK difference of only `+0.0016`. DenseNet is better on Accuracy, macro F1, Grade 1 recall, AP, and AUC. The QWK confidence intervals substantially overlap, so the SE-ResNeXt result is not evidence of a meaningful ordinal-performance improvement. DenseNet remains the more defensible application model; SE-ResNeXt remains the localization-oriented comparison model.
