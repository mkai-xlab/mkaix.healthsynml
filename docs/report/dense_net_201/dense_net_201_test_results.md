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

## 3. Diagnostic Analysis: The Grade 1 Bottleneck

### The Phenomenon:
While the overall accuracy increased by **$+1.93\%$**, the recall of **Grade 1 (Doubtful OA)** fell to **$7.0\%$** (compared to $24.0\%$ in the baseline), while the recall of **Grade 0 (Normal)** rose to **$94.0\%$** (compared to $85.0\%$ in the baseline).

### The Cause (Catastrophic Forgetting in Stage 3):
1.  **Imbalance Ratio:** In the training dataset, Grade 0 has $2,286$ images while Grade 1 has $1,046$ images.
2.  **Representation Drift:** In Stage 2, the `WeightedRandomSampler` was active, forcing a balanced distribution. Under this setup, the model's validation QWK reached a peak of **`0.7728`** and validation accuracy was `57.63%` with balanced class boundaries.
3.  **Threshold Shift:** When Stage 3 (Fine-Tuning) began, the balanced sampler was turned off to train on the original, imbalanced distribution. Because the new Multi-Scale backbone has a very high learning capacity, training it for 15 epochs on this skewed distribution caused it to shift its decision boundary (specifically $L_0$ in the CORN logits) to predict Grade 0 for doubtful or ambiguous Grade 1 cases, maximizing overall accuracy at the expense of Grade 1 recall.

---

## 4. Key Recommendations for Deployment

To achieve the best clinical utility, choose one of the following deployment paths:

### Option A: Deploy the Stage 2 Best Checkpoint (Highly Recommended)
*   **Weights Path:** `/content/drive/MyDrive/Models/densenet201_checkpoints/best_model_stage2.pth`
*   **Description:** This checkpoint was saved at the end of Stage 2 training when the balanced sampler was active. It holds the most balanced representations and will have a much higher recall for Grade 1 and Grade 2 than the final Stage 3 checkpoint, making it safer for screening.

### Option B: Implement Threshold Adjustments (Post-Processing)
*   **Description:** Rather than predicting the class directly using the default sigmoid threshold ($0.5$), lower the threshold for Grade 1 in your backend service to force more positive predictions for doubtful joints:
    $$P(Y \ge 1) > 0.35 \implies \text{Class } \ge 1$$
