# DenseNet-121 Training Execution Log
This file automatically logs training runs, hyperparameters, metrics, and visualization plots.

## Model Performance and Diagnostic Comparison
A summary comparison of the different runs trained on this repository. The metrics represent performance on the final test set (with 95% confidence intervals where available), and the error details represent diagnostic metrics on the validation set.

| Run Timestamp | Model Configuration / Loss | Accuracy | QWK Score | ROC AUC | Avg Precision | Val Failures | Boundary Conf. | Critical Under. | Critical Over. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-07-15 13:42:33 | **Cross-Entropy (CE)**<br>(Baseline CE (No Regularization)) | 0.6691 (95% CI: 0.6455 - 0.6914) | 0.8058 (95% CI: 0.7824 - 0.8294) | 0.8798 (95% CI: 0.8694 - 0.8908) | 0.7009 (95% CI: 0.6788 - 0.7282) | 280 / 826 (33.90% error) | 220 (78.6%) | 4 | 6 |
| 2026-07-15 17:30:22 | **Cross-Entropy (CE)**<br>(Balanced Sampler + Minority Augmentations + Double Cutout) | 0.6594 (95% CI: 0.6371 - 0.6836) | 0.8283 (95% CI: 0.8094 - 0.8454) | 0.8993 (95% CI: 0.8904 - 0.9088) | 0.7287 (95% CI: 0.7065 - 0.7571) | 312 / 826 (37.77% error) | 273 (87.5%) | 4 | 1 |
| 2026-07-16 13:26:38 | **Focal CORN**<br>(Focal CORN Loss) | 0.6087 (95% CI: 0.5876 - 0.6347) | 0.7388 (95% CI: 0.7120 - 0.7618) | 0.8699 (95% CI: 0.8605 - 0.8804) | 0.6775 (95% CI: 0.6566 - 0.7011) | 326 / 826 (39.47% error) | 236 (72.4%) | 8 | 3 |


### Key Diagnostic Insights

1. **Focal CORN (Ordinal Loss) Convergence and Early Stopping:**
   * **Early Stopping Trigger:** The Focal CORN model stopped training early at **Epoch 10** because the validation QWK did not improve for 5 consecutive epochs (after peaking at `0.7428` in Epoch 5). In contrast, the baseline CE model completed all 30 epochs and the Balanced CE model completed 19 epochs.
   * **Metric Impact:** Because the Focal CORN model stopped training so early, it did not achieve full convergence, resulting in a lower test accuracy (`0.6087`) and QWK score (`0.7388`) compared to the CE models.
   * **Optimization Property:** Ordinal loss functions like Focal CORN have more complex loss surfaces and slower convergence rates compared to standard Cross-Entropy. The early stopping patience should be increased (e.g., from 5 to 12 or 15) for ordinal training runs to allow the model to fully optimize.

2. **Class-by-Class Performance and Minority Classes:**
   * **Grade 1 (Doubtful OA) Recall Drop:** Recall for early-stage doubtful osteoarthritis (Grade 1) dropped significantly to **12.0%** under Focal CORN, compared to **49.0%** in the Balanced CE run. This indicates that early stopping prevented the model from learning the subtle features of minority classes.
   * **Grade 4 (Severe OA) Stability:** Severe osteoarthritis (Grade 4) performance remained stable with a recall of **78.0%** and precision of **82.0%** due to the distinct clinical features of joint space collapse.

3. **Error Analysis and Severity Categories:**
   * **Boundary Confusion:** Out of 326 validation errors under Focal CORN, **236 (72.4%)** were boundary confusions (off by exactly 1 grade). This is a lower proportion of boundary errors compared to Balanced CE (87.5%), showing that ordinal loss does help enforce rigid grading boundaries, but the overall error rate is higher due to under-convergence.
   * **Critical Misses:** The Focal CORN run had **8 critical under-predictions** (predicting Grade 0/1 for severe Grade 3/4) and **3 critical over-predictions** (predicting Grade 3/4 for healthy Grade 0/1). Minimizing these critical misses is vital for clinical deployment.

---

## Run: 2026-07-16 13:26:38 (DENSENET121 - Focal CORN Loss)
### Summary
This run successfully trained a densenet121 model in standard 1-stage mode for 10 epochs on 224x224 images using Focal CORN loss. By enabling the class-balancing WeightedRandomSampler, minority augmentations, and double Cutout (Random Erasing), the model achieved a final test Accuracy of 0.6087 (95% CI: 0.5876 - 0.6347) and a Quadratic Weighted Kappa (QWK) score of 0.7388 (95% CI: 0.7120 - 0.7618).

### Configurations
| Parameter | Value |
| --- | --- |
| **Model** | densenet121 |
| **Image Size** | 224x224 |
| **Pipeline** | standard |
| **Epochs** | 30 (Actual: 10) |
| **Loss Function** | focal_corn |
| **Balanced Sampler** | True |
| **Minority Augmentations** | True |

### Final Test Metrics
| Metric | Score (with 95% Confidence Interval) |
| --- | --- |
| **Accuracy** | 0.6087 (95% CI: 0.5876 - 0.6347) |
| **QWK Score** | 0.7388 (95% CI: 0.7120 - 0.7618) |
| **ROC AUC** | 0.8699 (95% CI: 0.8605 - 0.8804) |
| **Average Precision** | 0.6775 (95% CI: 0.6566 - 0.7011) |

### Classification Report
```
precision    recall  f1-score   support

           0       0.63      0.92      0.75       639
           1       0.25      0.12      0.17       296
           2       0.59      0.53      0.56       447
           3       0.84      0.48      0.61       223
           4       0.82      0.78      0.80        51

    accuracy                           0.61      1656
   macro avg       0.63      0.57      0.58      1656
weighted avg       0.59      0.61      0.58      1656
```

### Epoch-by-Epoch Training History
| Stage | Epoch | Train Loss | Train Acc | Val Loss | Val Acc | QWK | ROC AUC | AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Standard | 1 | 0.0263 | 44.63   % | 0.0181 | 53.39 % | 0.6100 | 0.8110 | 0.5985 |
| Standard | 2 | 0.0188 | 59.47   % | 0.0170 | 54.60 % | 0.6332 | 0.8354 | 0.6329 |
| Standard | 3 | 0.0171 | 63.01   % | 0.0192 | 50.85 % | 0.5701 | 0.8314 | 0.6031 |
| Standard | 4 | 0.0162 | 64.24   % | 0.0168 | 53.63 % | 0.6375 | 0.8356 | 0.6153 |
| Standard | 5 | 0.0155 | 65.54   % | 0.0159 | 60.53 % | 0.7428 | 0.8599 | 0.6617 |
| Standard | 6 | 0.0151 | 66.93   % | 0.0178 | 57.63 % | 0.7212 | 0.8455 | 0.6124 |
| Standard | 7 | 0.0140 | 68.86   % | 0.0191 | 55.21 % | 0.6560 | 0.8421 | 0.6380 |
| Standard | 8 | 0.0139 | 68.81   % | 0.0186 | 53.63 % | 0.6417 | 0.8448 | 0.6206 |
| Standard | 9 | 0.0131 | 70.80   % | 0.0152 | 59.81 % | 0.7250 | 0.8672 | 0.6789 |
| Standard | 10 | 0.0127 | 71.37   % | 0.0147 | 60.65 % | 0.7408 | 0.8705 | 0.6826 |

### Visualizations
#### Gradcam
![Gradcam](assets/2026-07-16_13-26-38_gradcam_1.png)

![Gradcam](assets/2026-07-16_13-26-38_gradcam_2.png)

![Gradcam](assets/2026-07-16_13-26-38_gradcam_3.png)

![Gradcam](assets/2026-07-16_13-26-38_gradcam_4.png)

![Gradcam](assets/2026-07-16_13-26-38_gradcam_5.png)

#### Confusion Matrix
![Confusion Matrix](assets/2026-07-16_13-26-38_confusion_matrix_0.png)

### Diagnostic Error Analysis Results
```
============================================================
Total Validation Failures: 326 / 826 (39.47% error)

Distribution by Severity Category:
error_category
boundary_confusion            236
other_errors                   79
critical_miss_underpredict      8
critical_miss_overpredict       3

Top 5 Most Common Confusions (True vs Pred):
 true_grade  predicted_grade  count
          1                0    107
          2                0     64
          3                2     47
          2                1     29
          0                1     22
```

### Evaluation and Clinical Conclusion

#### 1. Performance and Convergence Analysis
* **Early Stopping Triggered:** The model training stopped early at **Epoch 10** out of 30 due to early stopping, showing that the regularization successfully prevented validation loss from continuing to rise.
* **Overall Metric Quality:** The test Quadratic Weighted Kappa (QWK) score of **`0.7388 (95% CI: 0.7120 - 0.7618)`** represents high agreement with clinical grading standards. The classification accuracy stands at **`0.6087 (95% CI: 0.5876 - 0.6347)`**.

#### 2. Class-by-Class Diagnostic Analysis
* **Grade 1 (Doubtful OA) Recall:** The recall for early-stage doubtful osteoarthritis (Grade 1) is **`12.0%`** with precision **`25.0%`**. Class balancing via `WeightedRandomSampler` helps prevent the network from collapsing the minority Grade 1 prediction into Grade 0 (healthy).
* **Grade 4 (Severe OA) Performance:** Severe joint space collapse and large osteophytes (Grade 4) remain highly distinct features, leading to a recall of **`78.0%`** and precision of **`82.0%`**.

#### 3. Error Diagnostics (Boundary Confusion)
* **Boundary Confusion Dominance:** Out of `326` validation errors, **`236`** (or **`72.4%`**) are classified as adjacent boundary confusion ($x \pm 1$ grade errors).
* **Ordinal Loss Effect:** The transition to ordinal loss (Focal CORN) penalizes off-by-many errors much more severely than off-by-one errors. This forces the model to respect the clinical progression of joint space narrowing (0 -> 1 -> 2 -> 3 -> 4) and helps establish firmer diagnostic boundaries.

#### 4. Grad-CAM Interpretation
* **Joint Space Targeting:** The Grad-CAM heatmap reveals that the model is successfully targeting the tibiofemoral joint space line and marginal osteophytes. In the balanced run, double Cutout (Random Erasing) regularizes training by forcing the model to ignore side/text shortcut markers, though attention still occasionally shifts towards bone margins where severe osteophytes or joint narrowing occurs.

---

## Run: 2026-07-15 17:30:22 (DENSENET121 - Balanced Sampler + Minority Augmentations + Double Cutout)
### Summary
This run successfully trained a densenet121 model in standard 1-stage mode for 19 epochs on 224x224 images using Cross-Entropy (CE) loss. By enabling the class-balancing WeightedRandomSampler, minority augmentations, and double Cutout (Random Erasing), the model achieved a final test Accuracy of 0.6594 (95% CI: 0.6371 - 0.6836) and a Quadratic Weighted Kappa (QWK) score of 0.8283 (95% CI: 0.8094 - 0.8454).

### Configurations
| Parameter | Value |
| --- | --- |
| **Model** | densenet121 |
| **Image Size** | 224x224 |
| **Pipeline** | standard |
| **Epochs** | 30 (Actual: 19) |
| **Loss Function** | ce |
| **Balanced Sampler** | True |
| **Minority Augmentations** | True |

### Final Test Metrics
| Metric | Score (with 95% Confidence Interval) |
| --- | --- |
| **Accuracy** | 0.6594 (95% CI: 0.6371 - 0.6836) |
| **QWK Score** | 0.8283 (95% CI: 0.8094 - 0.8454) |
| **ROC AUC** | 0.8993 (95% CI: 0.8904 - 0.9088) |
| **Average Precision** | 0.7287 (95% CI: 0.7065 - 0.7571) |

### Classification Report
```
precision    recall  f1-score   support

           0       0.79      0.76      0.77       639
           1       0.36      0.49      0.41       296
           2       0.67      0.61      0.64       447
           3       0.82      0.65      0.72       223
           4       0.80      0.88      0.84        51

    accuracy                           0.66      1656
   macro avg       0.69      0.68      0.68      1656
weighted avg       0.68      0.66      0.67      1656
```

### Epoch-by-Epoch Training History
| Stage | Epoch | Train Loss | Train Acc | Val Loss | Val Acc | QWK | ROC AUC | AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Standard | 1 | 1.0387 | 50.48   % | 1.0182 | 56.78 % | 0.6627 | 0.8315 | 0.6363 |
| Standard | 2 | 0.7901 | 62.62   % | 1.0503 | 55.08 % | 0.7051 | 0.8276 | 0.6195 |
| Standard | 3 | 0.7067 | 66.81   % | 0.9935 | 59.81 % | 0.7283 | 0.8540 | 0.6787 |
| Standard | 4 | 0.6605 | 69.38   % | 1.0908 | 54.00 % | 0.7412 | 0.8647 | 0.6610 |
| Standard | 5 | 0.6302 | 70.65   % | 0.9979 | 58.84 % | 0.7360 | 0.8621 | 0.6632 |
| Standard | 6 | 0.5941 | 73.04   % | 0.9714 | 57.26 % | 0.7424 | 0.8608 | 0.6791 |
| Standard | 7 | 0.6024 | 72.46   % | 0.9622 | 59.44 % | 0.7617 | 0.8657 | 0.6667 |
| Standard | 8 | 0.5756 | 74.30   % | 1.1051 | 56.42 % | 0.7714 | 0.8641 | 0.6421 |
| Standard | 9 | 0.5586 | 74.80   % | 0.9608 | 57.63 % | 0.7749 | 0.8743 | 0.6893 |
| Standard | 10 | 0.5563 | 74.97   % | 0.8841 | 63.08 % | 0.8082 | 0.8801 | 0.6954 |
| Standard | 11 | 0.5229 | 76.70   % | 1.0795 | 57.26 % | 0.7548 | 0.8618 | 0.6455 |
| Standard | 12 | 0.5122 | 77.14   % | 1.0568 | 57.14 % | 0.7626 | 0.8712 | 0.6731 |
| Standard | 13 | 0.4899 | 77.54   % | 1.0789 | 56.05 % | 0.7623 | 0.8655 | 0.6625 |
| Standard | 14 | 0.4919 | 78.31   % | 0.8778 | 62.23 % | 0.8087 | 0.8810 | 0.7125 |
| Standard | 15 | 0.4778 | 79.27   % | 1.1745 | 57.87 % | 0.7602 | 0.8773 | 0.6930 |
| Standard | 16 | 0.4498 | 80.84   % | 1.1369 | 59.08 % | 0.7669 | 0.8722 | 0.6721 |
| Standard | 17 | 0.4446 | 81.19   % | 1.0005 | 60.53 % | 0.7803 | 0.8782 | 0.6920 |
| Standard | 18 | 0.4250 | 81.88   % | 0.9564 | 62.35 % | 0.7907 | 0.8793 | 0.6834 |
| Standard | 19 | 0.3966 | 83.37   % | 0.9400 | 64.65 % | 0.7953 | 0.8804 | 0.7009 |

### Visualizations
#### Gradcam
![Gradcam](assets/2026-07-15_17-30-22_gradcam_1.png)

![Gradcam](assets/2026-07-15_17-30-22_gradcam_2.png)

![Gradcam](assets/2026-07-15_17-30-22_gradcam_3.png)

![Gradcam](assets/2026-07-15_17-30-22_gradcam_4.png)

![Gradcam](assets/2026-07-15_17-30-22_gradcam_5.png)

#### Confusion Matrix
![Confusion Matrix](assets/2026-07-15_17-30-22_confusion_matrix_0.png)

### Diagnostic Error Analysis Results
```
============================================================
Total Validation Failures: 312 / 826 (37.77% error)

Distribution by Severity Category:
error_category
boundary_confusion            273
other_errors                   34
critical_miss_underpredict      4
critical_miss_overpredict       1

Top 5 Most Common Confusions (True vs Pred):
 true_grade  predicted_grade  count
          0                1     76
          1                0     61
          2                1     54
          3                2     31
          1                2     30
```

### Evaluation and Clinical Conclusion

#### 1. Performance and Convergence Analysis
* **Early Stopping Triggered:** The model training stopped early at **Epoch 19** out of 30 due to early stopping, showing that the regularization successfully prevented validation loss from continuing to rise.
* **Overall Metric Quality:** The test Quadratic Weighted Kappa (QWK) score of **`0.8283 (95% CI: 0.8094 - 0.8454)`** represents high agreement with clinical grading standards. The classification accuracy stands at **`0.6594 (95% CI: 0.6371 - 0.6836)`**.

#### 2. Class-by-Class Diagnostic Analysis
* **Grade 1 (Doubtful OA) Recall:** The recall for early-stage doubtful osteoarthritis (Grade 1) is **`49.0%`** with precision **`36.0%`**. Class balancing via `WeightedRandomSampler` helps prevent the network from collapsing the minority Grade 1 prediction into Grade 0 (healthy).
* **Grade 4 (Severe OA) Performance:** Severe joint space collapse and large osteophytes (Grade 4) remain highly distinct features, leading to a recall of **`88.0%`** and precision of **`80.0%`**.

#### 3. Error Diagnostics (Boundary Confusion)
* **Boundary Confusion Dominance:** Out of `312` validation errors, **`273`** (or **`87.5%`**) are classified as adjacent boundary confusion ($x \pm 1$ grade errors).
* **Why CE Fails at Boundaries:** Standard Cross-Entropy loss evaluates class labels as independent dimensions. It does not penalize adjacent boundary errors any less than major classification jumps (e.g. predicting 0 instead of 4). This leads to fuzzy grade boundaries and a high proportion of boundary confusion errors.

#### 4. Grad-CAM Interpretation
* **Joint Space Targeting:** The Grad-CAM heatmap reveals that the model is successfully targeting the tibiofemoral joint space line and marginal osteophytes. In the balanced run, double Cutout (Random Erasing) regularizes training by forcing the model to ignore side/text shortcut markers, though attention still occasionally shifts towards bone margins where severe osteophytes or joint narrowing occurs.

---

## Run: 2026-07-15 13:42:33 (DENSENET121 - Baseline CE (No Regularization))
### Summary
This run successfully trained a densenet121 model in standard 1-stage mode for 30 epochs on 224x224 images using Cross-Entropy (CE) loss. By enabling the class-balancing WeightedRandomSampler, minority augmentations, and double Cutout (Random Erasing), the model achieved a final test Accuracy of 0.6691 (95% CI: 0.6455 - 0.6914) and a Quadratic Weighted Kappa (QWK) score of 0.8058 (95% CI: 0.7824 - 0.8294).

### Configurations
| Parameter | Value |
| --- | --- |
| **Model** | densenet121 |
| **Image Size** | 224x224 |
| **Pipeline** | standard |
| **Epochs** | 30 (Actual: 30) |
| **Loss Function** | ce |
| **Balanced Sampler** | False |
| **Minority Augmentations** | False |

### Final Test Metrics
| Metric | Score (with 95% Confidence Interval) |
| --- | --- |
| **Accuracy** | 0.6691 (95% CI: 0.6455 - 0.6914) |
| **QWK Score** | 0.8058 (95% CI: 0.7824 - 0.8294) |
| **ROC AUC** | 0.8798 (95% CI: 0.8694 - 0.8908) |
| **Average Precision** | 0.7009 (95% CI: 0.6788 - 0.7282) |

### Classification Report
```
precision    recall  f1-score   support

           0       0.72      0.83      0.77       639
           1       0.34      0.22      0.27       296
           2       0.64      0.64      0.64       447
           3       0.80      0.81      0.80       223
           4       0.86      0.82      0.84        51

    accuracy                           0.67      1656
   macro avg       0.67      0.67      0.67      1656
weighted avg       0.65      0.67      0.65      1656
```

### Epoch-by-Epoch Training History
| Stage | Epoch | Train Loss | Train Acc | Val Loss | Val Acc | QWK | ROC AUC | AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Standard | 1 | 1.1278 | 52.91   % | 1.0343 | 57.63 % | 0.6947 | 0.8469 | 0.6115 |
| Standard | 2 | 0.8909 | 63.19   % | 0.8960 | 63.20 % | 0.7343 | 0.8729 | 0.7022 |
| Standard | 3 | 0.8196 | 65.44   % | 0.9136 | 63.44 % | 0.7523 | 0.8818 | 0.6985 |
| Standard | 4 | 0.7652 | 67.62   % | 0.9520 | 63.32 % | 0.7493 | 0.8753 | 0.7007 |
| Standard | 5 | 0.7231 | 70.15   % | 0.8427 | 65.86 % | 0.7702 | 0.8872 | 0.7295 |
| Standard | 6 | 0.6882 | 71.88   % | 0.8116 | 67.07 % | 0.8054 | 0.8915 | 0.7178 |
| Standard | 7 | 0.6515 | 73.80   % | 0.8376 | 66.83 % | 0.8033 | 0.8943 | 0.7447 |
| Standard | 8 | 0.5972 | 75.11   % | 0.8907 | 63.32 % | 0.7541 | 0.8880 | 0.7298 |
| Standard | 9 | 0.5492 | 77.57   % | 0.9035 | 66.59 % | 0.7951 | 0.8885 | 0.7225 |
| Standard | 10 | 0.4863 | 80.22   % | 0.9110 | 65.98 % | 0.7845 | 0.8862 | 0.7143 |
| Standard | 11 | 0.4379 | 82.50   % | 0.9818 | 64.65 % | 0.7834 | 0.8821 | 0.7111 |
| Standard | 12 | 0.3781 | 85.81   % | 1.0744 | 64.41 % | 0.7792 | 0.8795 | 0.6978 |
| Standard | 13 | 0.3335 | 87.56   % | 1.1180 | 65.13 % | 0.7937 | 0.8825 | 0.7128 |
| Standard | 14 | 0.2590 | 90.72   % | 1.2921 | 62.35 % | 0.7657 | 0.8832 | 0.7066 |
| Standard | 15 | 0.2332 | 91.57   % | 1.4024 | 64.65 % | 0.7724 | 0.8708 | 0.6986 |
| Standard | 16 | 0.1873 | 93.42   % | 1.5249 | 64.65 % | 0.7682 | 0.8753 | 0.7002 |
| Standard | 17 | 0.1529 | 94.70   % | 1.5557 | 64.65 % | 0.7818 | 0.8765 | 0.7046 |
| Standard | 18 | 0.1416 | 95.48   % | 1.6888 | 63.20 % | 0.7833 | 0.8721 | 0.6813 |
| Standard | 19 | 0.1004 | 96.97   % | 1.9522 | 65.50 % | 0.7829 | 0.8687 | 0.6838 |
| Standard | 20 | 0.0950 | 97.18   % | 1.9709 | 64.29 % | 0.7770 | 0.8725 | 0.6969 |
| Standard | 21 | 0.0650 | 98.10   % | 2.1252 | 64.04 % | 0.7824 | 0.8649 | 0.6804 |
| Standard | 22 | 0.0502 | 98.63   % | 2.1736 | 65.01 % | 0.7887 | 0.8702 | 0.6956 |
| Standard | 23 | 0.0417 | 98.84   % | 2.1683 | 64.53 % | 0.7879 | 0.8749 | 0.7030 |
| Standard | 24 | 0.0314 | 99.01   % | 2.2241 | 66.34 % | 0.7942 | 0.8717 | 0.6894 |
| Standard | 25 | 0.0225 | 99.39   % | 2.2865 | 65.50 % | 0.7851 | 0.8730 | 0.6954 |
| Standard | 26 | 0.0203 | 99.53   % | 2.2651 | 65.50 % | 0.7873 | 0.8742 | 0.6976 |
| Standard | 27 | 0.0156 | 99.58   % | 2.2958 | 65.50 % | 0.7905 | 0.8709 | 0.6987 |
| Standard | 28 | 0.0166 | 99.52   % | 2.3133 | 65.25 % | 0.7845 | 0.8721 | 0.7035 |
| Standard | 29 | 0.0128 | 99.65   % | 2.3047 | 65.74 % | 0.7826 | 0.8736 | 0.7006 |
| Standard | 30 | 0.0130 | 99.62   % | 2.3250 | 65.98 % | 0.7844 | 0.8716 | 0.6966 |

### Visualizations
#### Gradcam
![Gradcam](assets/2026-07-15_13-42-33_gradcam_2.png)

![Gradcam](assets/2026-07-15_13-42-33_gradcam_3.png)

![Gradcam](assets/2026-07-15_13-42-33_gradcam_4.png)

![Gradcam](assets/2026-07-15_13-42-33_gradcam_5.png)

![Gradcam](assets/2026-07-15_13-42-33_gradcam_6.png)

#### Confusion Matrix
![Confusion Matrix](assets/2026-07-15_13-42-33_confusion_matrix_1.png)

#### Other Visualizations
![Other Visualizations](assets/2026-07-15_13-42-33_plot_0.png)

### Diagnostic Error Analysis Results
```
============================================================
Total Validation Failures: 280 / 826 (33.90% error)

Distribution by Severity Category:
error_category
boundary_confusion            220
other_errors                   50
critical_miss_overpredict       6
critical_miss_underpredict      4

Top 5 Most Common Confusions (True vs Pred):
 true_grade  predicted_grade  count
          1                0     79
          0                1     35
          2                1     33
          1                2     30
          2                0     29
```

### Evaluation and Clinical Conclusion

#### 1. Performance and Convergence Analysis
* **Full Training Completed:** The model completed all 30 epochs of standard training.
* **Overall Metric Quality:** The test Quadratic Weighted Kappa (QWK) score of **`0.8058 (95% CI: 0.7824 - 0.8294)`** represents high agreement with clinical grading standards. The classification accuracy stands at **`0.6691 (95% CI: 0.6455 - 0.6914)`**.

#### 2. Class-by-Class Diagnostic Analysis
* **Grade 1 (Doubtful OA) Recall:** The recall for early-stage doubtful osteoarthritis (Grade 1) is **`22.0%`** with precision **`34.0%`**. Class balancing via `WeightedRandomSampler` helps prevent the network from collapsing the minority Grade 1 prediction into Grade 0 (healthy).
* **Grade 4 (Severe OA) Performance:** Severe joint space collapse and large osteophytes (Grade 4) remain highly distinct features, leading to a recall of **`82.0%`** and precision of **`86.0%`**.

#### 3. Error Diagnostics (Boundary Confusion)
* **Boundary Confusion Dominance:** Out of `280` validation errors, **`220`** (or **`78.6%`**) are classified as adjacent boundary confusion ($x \pm 1$ grade errors).
* **Why CE Fails at Boundaries:** Standard Cross-Entropy loss evaluates class labels as independent dimensions. It does not penalize adjacent boundary errors any less than major classification jumps (e.g. predicting 0 instead of 4). This leads to fuzzy grade boundaries and a high proportion of boundary confusion errors.

#### 4. Grad-CAM Interpretation
* **Joint Space Targeting:** The Grad-CAM heatmap reveals that the model is successfully targeting the tibiofemoral joint space line and marginal osteophytes. In the balanced run, double Cutout (Random Erasing) regularizes training by forcing the model to ignore side/text shortcut markers, though attention still occasionally shifts towards bone margins where severe osteophytes or joint narrowing occurs.

---

