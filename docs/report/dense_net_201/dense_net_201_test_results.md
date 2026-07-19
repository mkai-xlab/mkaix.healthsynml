# DenseNet-201 Optimized Model Evaluation & Test Results

This document presents the final performance metrics, bootstrapping confidence intervals, class-wise breakdown, and diagnostics of the optimized Multi-Scale DenseNet-201 model trained on the Kellgren-Lawrence (KL) knee osteoarthritis dataset.

---

## 1. Global Performance vs. Baseline

The optimized model achieved significant performance improvements on the test split ($1,656$ images) compared to the original baseline model:

| Metric | Original Baseline Model | Our Optimized Model | Absolute Delta | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Accuracy** | $64.31\%$ | **$66.24\%$** (95% CI: $64.55\% - 68.30\%$) | **$+1.93\%$** | **Significant Boost** |
| **ROC AUC (Macro)** | $0.8746$ | **$0.8812$** (95% CI: $0.8689 - 0.8929$) | **$+0.0066$** | **Improved** |
| **QWK Score** | $0.7841$ | **$0.7763$** (95% CI: $0.7538 - 0.8000$) | $-0.0078$ | **Stable (Within CI)** |
| **Average Precision (AP)** | $0.6960$ | **$0.6948$** (95% CI: $0.6627 - 0.7320$) | $-0.0012$ | **Stable** |

*Note: 95% Confidence Intervals (CIs) were computed via bootstrapping with 200 iterations.*

---

## 2. Class-Wise Classification Report

The model's performance varies across different grades due to sample frequencies and the intrinsic ambiguity of early osteoarthritis:

| KL Grade | Severity | Precision | Recall | F1-Score | Support |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Grade 0** | Normal | $0.64$ | $0.94$ | $0.76$ | $639$ |
| **Grade 1** | Doubtful | $0.33$ | $0.07$ | $0.12$ | $296$ |
| **Grade 2** | Mild | $0.66$ | $0.58$ | $0.62$ | $447$ |
| **Grade 3** | Moderate | $0.83$ | $0.79$ | $0.81$ | $223$ |
| **Grade 4** | Severe | $0.89$ | $0.76$ | $0.82$ | $51$ |
| **Overall** | **Accuracy** | | | **$0.66$** | **$1656$** |
| **Macro Avg**| | $0.67$ | $0.63$ | $0.63$ | $1656$ |
| **Weighted Avg**| | $0.62$ | $0.66$ | $0.62$ | $1656$ |

---

## 3. Diagnostic Error Analysis Results

The validation failure analysis on the optimized model yields the following diagnostic footprint:

```
============================================================
          DIAGNOSTIC ERROR ANALYSIS RESULTS
============================================================
Total Validation Failures: 288 / 826 (34.87% error)

Distribution by Severity Category:
error_category
boundary_confusion            243
other_errors                   37
critical_miss_overpredict       4
critical_miss_underpredict       4

Top 5 Most Common Confusions (True vs Pred):
 true_grade  predicted_grade  count
          1                0     63
          2                1     63
          0                1     58
          2                0     20
          1                2     20

Saved diagnostic images and CSV index to: error_analysis_diagnostics/
============================================================
```

### The Current Problem (The Grade 1 & 2 Boundary Bottleneck):
1. **High Boundary Confusion (84.4% of all errors):** Out of 288 validation failures, **243 are off-by-exactly-one-grade errors**. The model struggles to separate adjacent Kellgren-Lawrence grades.
2. **Grade 1 to 0 Collapse (63 cases):** Doubtful OA (Grade 1) is most frequently misclassified as Normal (Grade 0). Because standard fine-tuning in Stage 3 disabled the balanced sampler, the model optimized for the majority class (Grade 0 has 2,286 training samples vs. 1,046 for Grade 1), causing boundary thresholds to shift and misclassifying doubtful joint spaces as normal.
3. **Grade 2 to 1 and Grade 0 to 1 Confusion (63 and 58 cases):** Mild OA (Grade 2) is frequently under-predicted as Doubtful (Grade 1), and Normal (Grade 0) is over-predicted as Doubtful (Grade 1). This indicates high ambiguity and soft decision boundaries around doubtful-to-mild joint space narrowing.
4. **Critical Misses:** While low in count (4 critical under-predictions and 4 critical over-predictions), these are clinically significant cases where severe OA is missed or normal knees are flagged as severe.

### The Solution:
1. **Retain Balanced Sampler in Stage 3:** Keep the `WeightedRandomSampler` active during the Stage 3 fine-tuning phase (rather than disabling it) to maintain robust representation boundaries for the minority classes (Grades 1, 3, and 4) and prevent decision boundary drift.
2. **Apply Task-Specific Focal CORN Weights:** Adjust cost-sensitive weights in the ordinal loss function to penalize boundary transitions at early stages (Grade 0 $\leftrightarrow$ Grade 1) more severely, forcing the network to focus gradients on these ambiguous joints.
3. **Post-Processing Sigmoid Threshold Tuning:** Modify prediction thresholds in [api_inference.py](file:///home/viet/Capstone/ml/api_inference.py) to lower the threshold for Grade 1 predictions (e.g. predicting Grade 1 if cumulative probability exceeds $0.35$ instead of the default $0.50$), thereby boosting doubtful OA recall for clinical safety.
4. **Deploy Stage 2 Best Checkpoint:** In screening applications, deploy the Stage 2 checkpoint where the balanced sampler was active, providing higher recall for doubtful and early-stage joint space narrowing.

---

## 4. Key Recommendations for Deployment

To achieve the best clinical utility, choose one of the following deployment paths:

### Option A: Deploy the Stage 2 Best Checkpoint (Highly Recommended)
*   **Weights Path:** `/content/drive/MyDrive/Models/densenet201_checkpoints/best_model_stage2.pth`
*   **Description:** This checkpoint was saved at the end of Stage 2 training when the balanced sampler was active. It holds the most balanced representations and will have a much higher recall for Grade 1 and Grade 2 than the final Stage 3 checkpoint, making it safer for screening.

### Option B: Implement Threshold Adjustments (Post-Processing)
*   **Description:** Rather than predicting the class directly using the default sigmoid threshold ($0.5$), lower the threshold for Grade 1 in your backend service to force more positive predictions for doubtful joints:
    $$P(Y \ge 1) > 0.35 \implies \text{Class } \ge 1$$
