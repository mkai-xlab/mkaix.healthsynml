# KL Grade Classification from Knee X-Ray Using Deep CNN Ensemble

## Abstract

This study presents an ensemble of convolutional neural networks (CNNs) for automatic KL (Kellgren-Lawrence) grading of knee osteoarthritis from radiographic images. Our system combines DenseNet-121 and SE-ResNeXt-50 architectures trained with different loss functions and data augmentation strategies. The ensemble achieves competitive performance on the OAI dataset, with Quadratic Weighted Kappa (QWK) scores up to **0.8394** and **0.8216** for individual models.

## 1. Dataset and Preprocessing

### 1.1 Dataset
- **Source**: Osteoarthritis Initiative (OAI) dataset
- **Split**: Train/Validation/Test (hash-based deduplication, patient-level split)
  - Training: 5,778 unique images
  - Validation: 826 images
  - Test: 1,656 unique images
- **Classes**: KL Grades 0-4 (ordinal)

### 1.2 ROI Detection
- **Model**: YOLOv8n knee joint detector
- **Performance**: mAP50-95 = 0.8136, Precision = 0.9879, Recall = 0.9881
- **ROI Expansion**: 1.15× max(box width, height)

### 1.3 Preprocessing Pipeline
1. YOLO ROI crop with black padding
2. LAB CLAHE (α=1.25)
3. Square pad to 384×384
4. ToTensor + ImageNet normalization

## 2. Model Architectures

### 2.1 DenseNet-121
| Component | Specification |
|-----------|--------------|
| Architecture | DenseNet-121 with dense blocks |
| Growth Rate | 32 |
| Classifier | Standard linear head |
| Input Size | 224×224 or 384×384 |
| Parameters | ~8M |

### 2.2 SE-ResNeXt-50 32×4d
| Component | Specification |
|-----------|--------------|
| Architecture | SE-ResNeXt-50 with 32×4d cardinality |
| Backbone | ResNeXt bottleneck with Squeeze-Excitation |
| Classifier | Linear head |
| Input Size | 384×384 |
| Parameters | ~27M |

## 3. Training Configuration

### 3.1 DenseNet-121 Best Config (Focal CORN)
| Parameter | Value |
|-----------|-------|
| Loss | Focal CORN (ordinal) |
| Sampler | WeightedRandomSampler (balanced) |
| Epochs | 30 |
| Batch Size | 16 |
| Learning Rates | Warmup: 1e-3, Head: 1e-4, Backbone: 5e-5, Finetune: 2e-5 |
| Augmentation | H-flip (p=0.5), rotation ±8°, double Random Erasing (p=0.8-0.9) |
| Pipeline | 3-stage (warmup → coarse → finetune) |

### 3.2 SE-ResNeXt-50 Best Config
| Parameter | Value |
|-----------|-------|
| Loss | Cross-Entropy (CE) |
| Sampler | Full inverse-frequency |
| Epochs | 30 |
| Batch Size | 48 |
| Augmentation | H-flip (p=0.5), rotation ±5°, gamma (p=0.2), Random Erasing (p=0.1) |
| Pipeline | 3-stage with native CAM head |

## 4. Results

### 4.1 Individual Model Performance

| Model | Loss | Input | Test Acc | Test QWK | Macro F1 | Grade 1 Recall | ROC AUC |
|-------|------|-------|----------|----------|----------|-----------------|--------|
| **DenseNet-121 (Focal CORN)** | FocalCORN | 224×224 | **0.6733** | **0.8394** | 0.6900 | 0.44 | 0.8984 |
| DenseNet-121 (CE) | CE | 224×224 | 0.6691 | 0.8058 | 0.6700 | 0.22 | 0.8798 |
| DenseNet-121 (CORN) | CORN | 384×384 | 0.6715 | 0.8246 | 0.6800 | 0.31 | 0.8963 |
| **SE-ResNeXt-50 (Native CAM)** | CE | 384×384 | **0.6558** | **0.8216** | 0.6781 | 0.41 | 0.8980 |

### 4.2 Per-Grade Performance (DenseNet-121 Focal CORN)

| KL Grade | Support | Precision | Recall | F1-Score |
|----------|---------|-----------|--------|----------|
| Grade 0 | 639 | 0.72 | 0.76 | 0.74 |
| Grade 1 | 296 | 0.36 | 0.44 | 0.39 |
| Grade 2 | 447 | 0.69 | 0.67 | 0.68 |
| Grade 3 | 223 | 0.71 | 0.70 | 0.70 |
| Grade 4 | 51 | 0.78 | 0.67 | 0.72 |

### 4.3 Ensemble Strategy

The final ensemble combines DenseNet-121 and SE-ResNeXt-50 using **soft voting on raw logits**.

**How it works:**

1. **Logit extraction** — both models output a raw logits vector `z ∈ ℝ⁵` (one value per KL grade), without applying softmax individually
2. **Element-wise average** — `z_final = 0.5 × z_DN121 + 0.5 × z_SEResNeXt`
3. **Softmax** — `p_k = exp(z_k) / Σⱼ exp(z_j)` converts averaged logits to probabilities
4. **Argmax** — `class = argmax(p_k)` selects the predicted KL grade

**Why soft voting on logits works better than on probabilities:**

| Aspect | Soft voting on probabilities | Soft voting on logits (ours) |
|---|---|---|
| Temperature | Sensitive to calibration | Temperature-independent |
| Ordinal information | Preserved but diluted by softmax | Preserved — logits encode ordinal spacing |
| Robustness | Sensitive to individual model calibration | More robust — averaging before softmax |

**Why these two models complement each other:**

| Factor | DenseNet-121 | SE-ResNeXt-50 |
|---|---|---|
| Architecture | Dense connections (feature reuse) | ResNeXt bottlenecks + SE gating |
| Cardinality | — | 32×4d (groups) |
| Loss | Focal CORN (ordinal) | Cross-Entropy |
| Input size | 224×224 | 384×384 |
| Sampler | WeightedRandomSampler (balanced) | Full inverse-frequency |
| Inductive bias | Dense feature concatenation | Aggregated residual paths |

The **Focal CORN loss** in DenseNet-121 penalises adjacent-grade errors more heavily than distant errors, while SE-ResNeXt-50 with CE learns discriminative class boundaries independently. Averaging their logits combines these complementary decision boundaries.

**Result:** Ensemble QWK ~0.84 vs individual best 0.8394

## 5. Explainability

### 5.1 Grad-CAM Visualization
- Applied to DenseNet-121 backbone for disease localization
- Peak activation inside joint region: 0.99+
- Joint energy: 0.82, Border energy: 0.11

### 5.2 CAM Analysis
- Native CAM and Grad-CAM showed correlation of 1.0000
- No significant difference in localization quality
- Model focuses on relevant knee anatomy

## 6. Conclusion

This study demonstrates competitive KL grading performance using CNN ensembles:

1. **DenseNet-121 with Focal CORN** achieves best single-model QWK of 0.8394
2. **SE-ResNeXt-50** provides complementary predictions for ensemble
3. **Grade 1 (Doubtful OA)** remains the most challenging class
4. **Grad-CAM** provides interpretable disease localization

## Citation

```bibtex
@article{knee_oa_classification,
  title={KL Grade Classification from Knee X-Ray Using Deep CNN Ensemble},
  author={},
  year={2026}
}
```

## Source Data

Full experiment data available in: `report/report.csv`

## Diagrams

- [Inference Pipeline Diagram](../diagram/inference-pipeline-v2.drawio) - Open in draw.io desktop
- [DenseNet-121 Architecture](../diagram/densenet121.drawio)
- [SE-ResNeXt-50 Architecture](../diagram/resnext50.drawio)
- [Ensemble Architecture v4](../diagram/ensemble-v4.drawio) — **Pipeline** (page 1) + **Mechanism** (page 2)
  - PNG preview: [pipeline](../diagram/ensemble_pipeline.png) · [mechanism](../diagram/ensemble_mechanism.png)
