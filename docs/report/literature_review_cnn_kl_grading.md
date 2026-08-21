# Literature Review: CNN-based Knee Osteoarthritis KL Grading

## Overview

This report summarizes the best CNN models for knee osteoarthritis (KOA) grading using the Kellgren-Lawrence (KL) classification system from two peer-reviewed papers.

---

## Paper 1: Shahid et al. (2025) - Frontiers in Medicine

**Title:** Potential of AI-based diagnostic grading system for knee osteoarthritis

**DOI:** [10.3389/fmed.2025.1707588](https://doi.org/10.3389/fmed.2025.1707588)

### Best Model: DenseNet-121

| Aspect | Details |
|--------|---------|
| **Architecture** | DenseNet-121 (pretrained on ImageNet) + custom FC head |
| **Input Size** | 224×224 pixels |
| **Total Parameters** | 13,469,571 |
| **Trainable Parameters** | 13,385,923 |

### Dataset

| Property | Value |
|----------|-------|
| **Source** | Social Security Teaching Hospital, Lahore, Pakistan |
| **Total Images** | 602 knee images (301 patients) |
| **KL Distribution** | Grade 0: 18, Grade 1: 67, Grade 2: 105, Grade 3: 126, Grade 4: 86 |
| **Patient Demographics** | 162 female, 139 male; mean age 59.85 ± 11.4 years |
| **Age Range** | 40-80 years |
| **Train/Test Split** | 80% train, 20% test (61 samples) |
| **Validation Split** | 90% train, 10% validation (from augmented data) |

### Preprocessing Pipeline

1. **Knee Isolation:** Binary thresholding (OTSU) + morphological opening
2. **Cropping:** Column-wise detection of knee start/end points
3. **Padding:** Zero-padding to standardize dimensions
4. **Normalization:** Pixel intensities normalized to [0, 1]

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Framework** | TensorFlow + Keras |
| **Optimizer** | Adam |
| **Learning Rate** | Adaptive (with scheduler) |
| **Regularization** | L2 (λ = 1×10⁻⁴) + Dropout (rate = 0.3) |
| **Fine-tuning** | Full fine-tuning (no frozen layers) |
| **Hardware** | NVIDIA QUADRO P2000 (5GB), 64GB RAM |

### Data Augmentation

| Technique | Parameters |
|----------|------------|
| Rotation | ±30 degrees |
| Zoom | 0.8 - 1.2 |
| Horizontal Flip | Yes |
| Affine Shift | Up to 10% of image dimensions |
| Shear | ±20 degrees |

### Results (Test Set)

| Metric | Value |
|--------|-------|
| **Accuracy** | 68.85% |
| **AUC** | 85.67% |
| **Precision** | 68.33% |
| **Recall** | 67.21% |
| **Loss** | 2.382 |

### Comparison with Other Models

| Model | Accuracy | AUC | Precision | Recall |
|-------|----------|-----|-----------|--------|
| **DenseNet-121** | **68.85%** | **85.67%** | **68.33%** | **67.21%** |
| DenseNet-201 | 60.66% | 80.92% | 60.66% | 60.66% |
| MobileNet | 60.66% | 83.65% | 61.02% | 59.02% |
| ResNet50 | 60.66% | 83.03% | 60.00% | 59.02% |
| ResNet50-V2 | 54.10% | 79.91% | 55.93% | 54.10% |
| Inception-ResNetV2 | 55.37% | 80.70% | 58.77% | 55.37% |

---

## Paper 2: Tiulpin et al. (2019) - Scientific Reports

**Title:** Multimodal Machine Learning-based Knee osteoarthritis progression prediction from plain Radiographs and clinical Data

**DOI:** [10.1038/s41598-019-56527-3](https://doi.org/10.1038/s41598-019-56527-3)

### Best Model: SE-ResNeXt-50 32x4d (Multi-task CNN)

| Aspect | Details |
|--------|---------|
| **Architecture** | SE-ResNeXt-50 32x4d (pretrained on ImageNet) |
| **Head Type** | Multi-task: 2 FC branches |
| **Branch 1** | 3 outputs (progression: no/fast/slow) |
| **Branch 2** | 5 outputs (KL grade: 0-4) |
| **Input Size** | 310×310 pixels (ROI: 140×140mm) |

### Dataset

| Property | Training (OAI) | Testing (MOST) |
|----------|----------------|----------------|
| **Total Knees** | 4,928 (2,711 subjects) | 3,918 (2,129 subjects) |
| **Progressors** | 1,331 (27%) | 1,501 (47%) |
| **Baseline KL** | 0, 1, 2, 3 | 0, 1, 2, 3 |

### Progression Classes

- **y=0:** No knee OA progression
- **y=1:** Progression within 60 months (fast progression)
- **y=2:** Progression after 60 months (slow progression)

### Preprocessing Pipeline

1. **ROI Extraction:** 140×140mm using BoneFinder software
2. **Rotation:** Tibial plateau aligned horizontally
3. **Histogram Clipping:** 5th-99th percentiles
4. **Normalization:** Global contrast normalization
5. **Bit Depth:** Converted to 8-bit (×255)
6. **Resizing:** 310×310 pixels (0.45mm pixel spacing)
7. **Horizontal Flip:** Left knees flipped to match right

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Framework** | PyTorch 1.0 |
| **Optimizer** | Adam |
| **Learning Rate** | 1e-3 (dropped at epoch 15) |
| **Batch Size** | 64 |
| **Weight Decay** | 1e-4 |
| **Dropout** | 0.5 (before each FC layer) |
| **Epochs** | 2 frozen + 20 unfrozen |
| **Hardware** | 3× NVIDIA GTX 1080Ti |

### Data Augmentation (On-the-fly)

| Technique | Parameters |
|-----------|------------|
| Random Noise | Yes |
| Rotation | ±5 degrees |
| Random Cropping | 310×310 → 300×300 |
| Gamma Correction | Random |

### Test-Time Augmentation (TTA)

- 5-crop TTA (4 corners + center)
- Predictions averaged across all crops and CV models

### Results

#### CNN Only (Model 5)

| Metric | Cross-validation (OAI) | Test (MOST) |
|--------|----------------------|-------------|
| **AUC** | 0.76 | 0.79 (0.77-0.80) |
| **AP** | 0.56 | 0.68 (0.66-0.70) |

#### Multi-modal Fusion (Best Model)

| Model | AUC | AP |
|-------|-----|-----|
| CNN + Age, Sex, BMI, Injury, Surgery, WOMAC (Model 6) | 0.79 (0.78-0.81) | 0.68 (0.66-0.71) |
| **CNN + Clinical Data + KL-grade (Model 7)** | **0.81 (0.79-0.82)** | **0.70 (0.68-0.72)** |

#### Reference Methods Comparison

| Method | AUC | AP |
|--------|-----|-----|
| Logistic Regression (Age, Sex, BMI, KL) | 0.75 (0.74-0.77) | 0.62 (0.60-0.64) |
| GBM (Age, Sex, BMI, KL) | 0.76 (0.75-0.78) | 0.63 (0.61-0.65) |
| **Proposed CNN (Model 5)** | **0.79 (0.77-0.80)** | **0.68 (0.66-0.70)** |

#### KL-0/KL-1 Subgroup Results

| Model | AUC | AP |
|-------|-----|-----|
| CNN (Model 5) | 0.78 (0.76-0.80) | 0.58 (0.55-0.61) |
| **CNN + Clinical + KL (Model 7)** | **0.80 (0.78-0.82)** | **0.62 (0.58-0.65)** |

---

## Summary Comparison

| Aspect | Shahid et al. (2025) | Tiulpin et al. (2019) |
|--------|---------------------|----------------------|
| **Best CNN** | DenseNet-121 | SE-ResNeXt-50 32x4d |
| **Task** | KL Grading (5-class) | OA Progression + KL (multi-task) |
| **Input Size** | 224×224 | 310×310 |
| **Training Data** | 602 images | 4,928 knees |
| **Test Data** | 61 images | 3,918 knees |
| **Main Metric** | Accuracy: 68.85% | AUC: 0.79, AP: 0.68 |
| **Augmentation** | ±30° rotation, zoom, flip, shear | ±5° rotation, noise, gamma |
| **Multi-modal** | No | Yes (CNN + clinical data) |
| **Explainability** | Not reported | GradCAM attention maps |
| **Key Innovation** | Automated knee isolation | Multi-task learning + stacking |

---

## Key Findings

1. **Transfer learning** from ImageNet is essential for both approaches
2. **DenseNet-121** achieved best results for single-task KL grading (Paper 1)
3. **SE-ResNeXt-50** with multi-task learning excels in progression prediction (Paper 2)
4. **Multi-modal fusion** (CNN + clinical data) significantly improves performance
5. **Data augmentation** helps with limited medical imaging datasets
6. **GradCAM** provides interpretable attention maps for clinical trust

---

## References

1. Shahid S, et al. (2025). Potential of AI-based diagnostic grading system for knee osteoarthritis. *Frontiers in Medicine*, 12:1707588. doi: 10.3389/fmed.2025.1707588

2. Tiulpin A, et al. (2019). Multimodal Machine Learning-based Knee osteoarthritis progression prediction from plain Radiographs and clinical Data. *Scientific Reports*, 9:20038. doi: 10.1038/s41598-019-56527-3
