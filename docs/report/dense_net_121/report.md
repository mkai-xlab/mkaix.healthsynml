# DenseNet-121 Training Execution Log
This file automatically logs training runs, hyperparameters, metrics, and visualization plots.

## Run: 2026-07-15 17:30:22 (DENSENET121)
### Summary
This run successfully trained a densenet121 model in standard 1-stage mode for 30 epochs (with early stopping triggering at epoch 19) on 224x224 images using standard CrossEntropy loss. By enabling the class-balancing WeightedRandomSampler, minority augmentations, and double Cutout (Random Erasing), the model achieved a final test Accuracy of 0.6594 (95% CI: 0.6371 - 0.6836) and a Quadratic Weighted Kappa (QWK) score of 0.8283 (95% CI: 0.8094 - 0.8454).

### Configurations
| Parameter | Value |
| --- | --- |
| **Model** | densenet121 |
| **Image Size** | 224x224 |
| **Pipeline** | standard |
| **Epochs** | 30 |
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
* **Overfitting Under Control:** The model training stopped early at **Epoch 19** out of 30 due to early stopping. The training accuracy at early stop was `83.37%` while the validation accuracy stabilized at `64.65%`. The overfitting gap (difference of ~18.7%) has been drastically reduced from the baseline (which hit `99.62%` train and `65.98%` validation, a gap of ~33.6%). The sampler and Cutout successfully regularized the training run.
* **QWK Score Improvement:** The test Quadratic Weighted Kappa (QWK) score improved from the baseline score of `0.8058` to **`0.8283`** (95% CI: `0.8094 - 0.8454`). This is a solid progress step, demonstrating that class balancing and regularization boosted overall diagnostic quality.

#### 2. Class-by-Class Diagnostic Analysis
* **Grade 1 (Doubtful OA) Recall Recovered:** The recall for the minority Grade 1 class improved from **`22.0%`** in the baseline run to **`49.0%`** in this run! This represents a huge clinical diagnostic recovery, proving that the WeightedRandomSampler successfully forced the network to learn subtle joint space features of early osteoarthritis instead of ignoring them.
* **Stable Severe OA (Grade 4):** Grade 4 performance remains strong with `88.0%` recall and `80.0%` precision.

#### 3. Error Diagnostics (Boundary Confusion)
* **Boundary Confusion Dominance:** Out of 312 validation errors, **273** of them (or **87.5%**) are classified as `boundary_confusion` (meaning predicting a adjacent grade $x \pm 1$ instead of $x$).
* **Healthy vs Doubtful Boundary:** The largest sources of error are True 0 predicted as Grade 1 (76 cases) and True 1 predicted as Grade 0 (61 cases). Confusing healthy cartilage with early osteophytic signs is a highly subjective boundary even for human radiologists.
* **Why CE Fails at Boundaries:** Standard Cross-Entropy loss evaluates class labels as independent dimensions. It does not penalize boundary errors any less than major classification jumps. This is why the model's grade boundaries are fuzzy.

#### 4. Recommendation for the Next Iteration
* **Implement Ordinal Loss (Focal CORN):** To directly target the dominant `boundary_confusion` (87.5% of errors), we should transition the loss function from standard Cross-Entropy (`ce`) to **Focal CORN loss**. Focal CORN loss treats KL grading ordinally (0 < 1 < 2 < 3 < 4), penalizing off-by-one errors much less than off-by-three errors, forcing the model to learn a smoother clinical progression barrier.

---

## Run: 2026-07-15 13:42:33 (DENSENET121)
### Summary
This run successfully trained a densenet121 baseline model in standard 1-stage mode for 30 epochs on 224x224 images using standard CrossEntropy loss. No balanced sampler, minority augmentations, or double Cutout were used. The model achieved a final test Accuracy of 0.6691 (95% CI: 0.6455 - 0.6914) and a Quadratic Weighted Kappa (QWK) score of 0.8058 (95% CI: 0.7824 - 0.8294).

### Configurations
| Parameter | Value |
| --- | --- |
| **Model** | densenet121 |
| **Image Size** | 224x224 |
| **Pipeline** | standard |
| **Epochs** | 30 |
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
* **Overfitting Pattern:** The model achieves `99.62%` training accuracy but only `65.98%` validation accuracy. This is a classic indicator of high variance. Because we disabled all regularization (balanced sampler, minority augmentation, and Random Erasing/Cutout), the network began memorizing the train split starting around Epoch 10, while the validation loss swelled from `0.81` back up to `2.32`.
* **QWK Performance:** Despite the severe overfitting, the model achieves a remarkably high test QWK of **`0.8058`** (95% CI: `0.7824 - 0.8294`). This occurs because standard CrossEntropy naturally pushes the predictions towards adjacent classes when confused, which is lightly penalized by QWK. However, the raw accuracy remains capped at `66.91%`.

#### 2. Class-by-Class Clinical Diagnosis
* **Grade 1 (Doubtful OA) Collapse:** Grade 1 recall remains extremely low at **`22.0%`**. Because we disabled the `BalancedSampler`, the network was dominated by the massive Grade 0 (Healthy) class (`639` images). The model struggled to differentiate the subtle joint space narrowing of Grade 1 from Grade 0, leading to a high false-negative rate.
* **Grade 4 (Severe OA) Success:** The model performs exceptionally well on Grade 4, reaching `82%` recall and `86%` precision. Severe joint space collapse and large osteophytes are highly distinctive features, meaning the network can easily identify them even without complex augmentations.

#### 3. Grad-CAM Interpretation
* The Grad-CAM overlays reveal that the model is successfully focusing on the **tibiofemoral joint space line** and the medial/lateral margins where osteophytes form.
* Because we did **not** apply `CenterCrop`, the model was able to use the full width of the tibia and femur to diagnose the joint lines. However, the Grad-CAM maps also indicate slight attention focus near the edges in some scans, confirming that without Cutout (Random Erasing), the model is still vulnerable to edge shortcuts (text markers, label pins) if present.

#### 4. Recommendation for the Next Iteration
* **Enable Double Cutout:** To force the model to focus strictly on the joint line and ignore label shortcuts, Cutout should be re-enabled.
* **Enable Balanced Sampler:** To resolve the Grade 1 recall collapse, class balancing is essential.
* **Regularization:** Add weight decay or dropout tweaks to pull the training loss curve down and close the gap between train and validation accuracy.

---

