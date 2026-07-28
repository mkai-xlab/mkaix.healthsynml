# HealthSync: AI-Assisted Kellgren-Lawrence Grading of Knee Osteoarthritis

## Abstract

HealthSync is a web-based computer-aided analysis system for estimating the Kellgren-Lawrence (KL) grade of knee osteoarthritis from plain knee radiographs. The system detects individual knee regions with YOLOv8 and classifies each region into KL grades 0-4 using convolutional neural networks. The deployed service can return the predicted grade, class probabilities, detection box, processed knee image, and a native class-activation map (CAM). It is an assistive tool and is not a replacement for radiologist interpretation.

## 1. Introduction

Knee osteoarthritis (KOA) is a chronic joint disease associated with cartilage loss, joint-space narrowing, osteophytes, sclerosis, and bone deformation. Plain radiography is commonly used to assess these structural changes. The Kellgren-Lawrence scale provides five ordered grades:

- **Grade 0:** No definite radiographic abnormality.
- **Grade 1:** Doubtful narrowing or possible osteophytes.
- **Grade 2:** Definite osteophytes with possible joint-space narrowing.
- **Grade 3:** Multiple osteophytes, definite narrowing, and sclerosis.
- **Grade 4:** Severe narrowing, large osteophytes, marked sclerosis, and bone deformity.

Manual grading can be difficult, particularly at the boundaries between grades 0, 1, and 2. HealthSync is intended to provide a consistent second opinion from the radiograph while keeping the final decision with a qualified clinician.

### 1.1 Objectives

The project objectives are to:

1. Detect left and right knee regions in a bilateral radiograph.
2. Classify each detected knee into one of five KL grades.
3. Compare candidate CNN backbones and training configurations.
4. Provide a usable FastAPI inference service.
5. Provide visual evidence of the image region used by the classifier through native CAM.

### 1.2 Scope and limitations

The system uses radiographic image data only. It does not use age, sex, BMI, symptoms, WOMAC scores, laboratory results, or other clinical metadata. It estimates the current radiographic KL grade; it does not predict disease progression, treatment response, or future risk. It is not a fully autonomous diagnostic system.

## 2. Related Concepts

### 2.1 CNN classification

A CNN learns image features through convolutional layers and maps the extracted features to five class logits. Let $z_c$ be the logit for grade $c$, where $c \in \{0,1,2,3,4\}$. The softmax function converts logits into class probabilities:

\[
p(c \mid x) = \frac{e^{z_c}}{\sum_{j=0}^{4} e^{z_j}}
\]

The predicted grade is the class with the greatest probability:

\[
\hat{y} = \operatorname{argmax}_{c} p(c \mid x)
\]

### 2.2 Model architectures

- **DenseNet-121:** Each dense layer receives the feature maps produced by all earlier layers in the same dense block. This encourages feature reuse and supports the extraction of fine image details.
- **SE-ResNeXt-50 32x4d:** ResNeXt grouped convolutions learn several feature transformations in parallel. The squeeze-and-excitation module recalibrates feature channels, while residual connections help optimization.
- **EfficientNet-B0:** A compact CNN available as a standalone comparison mode.

The models are initialized with ImageNet-pretrained weights and adapted to output five KL grades. The project evaluates these models experimentally; a model is not promoted to production only because it has a high validation score.

## 3. Dataset and Data Preparation

### 3.1 Classification data

The classification experiments use the public Knee Osteoarthritis Dataset with Severity, derived from the Osteoarthritis Initiative (OAI) cohort. The dataset contains pre-cropped knee radiographs labeled with KL grades 0-4. The commonly reported class counts are:

| Grade | Description | Images |
|---|---|---:|
| 0 | Normal | 3,857 |
| 1 | Doubtful/minimal | 1,762 |
| 2 | Mild | 2,578 |
| 3 | Moderate | 1,286 |
| 4 | Severe | 267 |
| **Total** |  | **9,750** |

The data are imbalanced, with Grade 4 strongly under-represented. The experiments use a fixed train/validation/test organization. The consolidated experiments report 5,778 training images, 826 validation images, and 1,656 test images for the principal final runs. Patient-level grouping is not documented in the historical Kaggle split, so performance should not be interpreted as external clinical validation.

### 3.2 ROI detection data

A separate set of 1,500 bilateral radiographs was manually annotated with left and right knee bounding boxes for YOLOv8 training. The intended split was 80% training, 10% validation, and 10% testing: 1,200, 150, and 150 images respectively. The detector supplies the bounding boxes used by the inference pipeline; the detector and classifier should therefore be evaluated separately.

### 3.3 Preprocessing

The deployed classification preprocessing is:

1. Read the uploaded PNG or JPEG radiograph.
2. Detect knee bounding boxes with YOLOv8.
3. Extract each detected knee ROI.
4. Apply the configured laterality/orientation policy. The production candidate preserves natural left/right orientation; older checkpoints may use right-knee mirroring, so checkpoint metadata and preprocessing must agree.
5. Apply square padding so the complete ROI is retained without a center crop.
6. Apply LAB-space CLAHE to improve local contrast.
7. Resize the complete ROI to 384 x 384 pixels.
8. Convert to a tensor and apply ImageNet normalization.

Otsu thresholding is not part of the deployed preprocessing path. Training-only augmentation is applied according to the selected experiment, typically mild rotation, horizontal flipping, brightness/contrast jitter, and limited random erasing. Validation and test transformations are deterministic.

## 4. Training Method

The principal final DenseNet experiment uses three stages: five epochs of classification-head warm-up, fifteen epochs of coarse fine-tuning, and ten epochs of full fine-tuning. The models use AdamW, cosine learning-rate schedules, automatic mixed precision when available, and gradient-norm clipping. A full inverse-frequency weighted sampler is used to reduce the effect of class imbalance.

The main production classification loss is standard cross-entropy:

\[
\mathcal{L}_{CE} = -\sum_{c=0}^{4} y_c \log p(c \mid x)
\]

where $y_c$ is 1 for the true grade and 0 otherwise. Ordinal losses, focal losses, alternative samplers, augmentation policies, and CAM configurations were explored in ablation studies. They should be described as experiments, not as the deployed method unless the corresponding checkpoint is explicitly selected.

The validation selection score used in the principal DenseNet experiment is a composite score:

\[
S = 0.55\,QWK + 0.30\,F1_{macro} + 0.15\,AP_{macro}
\]

The test set is used only after model selection. The test set must not be repeatedly used to choose configurations.

## 5. Evaluation Metrics

Because KL grades are ordered and the classes are imbalanced, no single metric is sufficient.

### 5.1 Accuracy

\[
Accuracy = \frac{\text{number of correct predictions}}{\text{number of test samples}}
\]

Accuracy is easy to understand but can hide poor performance on rare grades.

### 5.2 Quadratic weighted kappa

Quadratic weighted kappa (QWK) measures agreement while penalizing distant grade errors more strongly than adjacent errors:

\[
QWK = 1 - \frac{\sum_{i,j} w_{ij}O_{ij}}{\sum_{i,j} w_{ij}E_{ij}},
\qquad
w_{ij}=\frac{(i-j)^2}{(K-1)^2}
\]

Here, $O_{ij}$ is the observed confusion-matrix count, $E_{ij}$ is the expected count under independent ratings, and $K=5$. A higher QWK indicates better agreement with the ordered reference grades.

### 5.3 Precision, recall, and F1

For a grade treated as a one-versus-rest class:

\[
Precision = \frac{TP}{TP+FP},\qquad
Recall = \frac{TP}{TP+FN}
\]

\[
F1 = \frac{2 \times Precision \times Recall}{Precision + Recall}
\]

Macro F1 gives every grade equal weight and is important for assessing rare grades. A confusion matrix and per-grade report should accompany aggregate results.

### 5.4 Average precision and ROC AUC

One-versus-rest average precision (AP) and ROC AUC summarize ranking quality from the model probabilities. They are supplementary metrics and do not replace per-grade recall or calibration analysis.

## 6. Explainability

The deployed system uses **native CAM**, not a generic Grad-CAM implementation. The classifier contains a class-map head that produces one spatial map for each KL grade. For grade $c$, the logit is the spatial mean of its class map:

\[
z_c = \frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W} M_c(i,j)
\]

The positive part of the selected grade map is normalized and resized over the knee ROI to create the displayed heatmap. This gives a direct correspondence between the class score and the visualization. The heatmap shows where the model found supporting evidence; it does not prove that a specific osteophyte or narrowing is clinically present.

In ensemble mode, DenseNet-121 and SE-ResNeXt probabilities are combined before selecting the final grade:

\[
p_{ens} = 0.55p_{DenseNet} + 0.45p_{SE-ResNeXt}
\]

The selected CAM is also checked with an anatomy gate. The service prefers a map with sufficient joint-region energy, limited border/lower-tibia energy, and a peak inside the broad joint band. If no map passes, the system returns the best available map and emits a warning.

## 7. System Architecture and API

The application is implemented with FastAPI and can run locally or in Docker. The inference flow is:

```text
Uploaded radiograph
        |
        v
YOLOv8 knee detection
        |
        v
ROI extraction and preprocessing
        |
        v
Selected classifier or DenseNet/SE-ResNeXt ensemble
        |
        v
KL probabilities, grade, confidence, native CAM
        |
        v
JSON response and annotated radiograph
```

The main endpoints are:

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/health` | Service readiness check |
| `GET /api/v1/models` | Model and configuration information |
| `POST /api/v1/predict` | Detect, classify, and return CAM results |
| `POST /api/v1/predict/detect-roi` | Return ROI detections without classification |

`MODEL_MODE` supports `densenet121`, `se_resnext`, `efficientnet_b0`, and `ensemble`. The service validates required checkpoint files and architecture metadata at startup and does not silently use random weights.

## 8. Principal Results

The final natural-orientation DenseNet-121 cross-entropy run selected epoch 24 using validation data. On the reported 1,656-image test set it achieved:

| Metric | Result |
|---|---:|
| Accuracy | 0.6504 |
| QWK | 0.8197 |
| Macro F1 | 0.6823 |
| Average Precision | 0.7309 |
| ROC AUC | 0.8935 |

The preferred natural-orientation SE-ResNeXt experiment reported accuracy 0.6558, QWK 0.8216, macro F1 0.6730, average precision 0.7299, and ROC AUC 0.8980 on the same reported test size. These results are internal benchmark results, not evidence of clinical deployment performance. The final production model should be identified by checkpoint path, architecture metadata, preprocessing version, and experiment timestamp.

## 9. Safety, Limitations, and Reproducibility

- The model can fail when the radiograph is outside the training distribution, poorly positioned, low quality, or missing a complete knee.
- Grade 1 and other boundary cases are intrinsically difficult, and the rare Grade 4 class requires separate attention to recall.
- YOLO detection errors can propagate into classification and CAM quality.
- CAM is an explanation aid, not a segmentation, measurement, or diagnosis.
- Reported benchmark data come from a public dataset and historical split; independent external validation and documented patient-level separation are still needed.
- No patient data, secrets, or checkpoint binaries should be committed to the repository. Checkpoints are mounted read-only for deployment.

For reproducibility, every promoted run should archive its exact code/configuration, random seed, dataset split, checkpoint, preprocessing, metrics, figures, and timestamp. Validation results are used for configuration decisions; the locked test set is opened only for final comparison.

## 10. Conclusion

HealthSync combines object detection, image preprocessing, CNN-based five-class KL grading, native CAM, and a FastAPI deployment into one reproducible inference workflow. Its main contribution is an end-to-end assistive system that can process a bilateral radiograph and return separate knee-level predictions with visual evidence. Further work should focus on patient-level and external validation, detector evaluation, calibration, subgroup analysis, and prospective assessment with radiologists before clinical use.

## References

1. Kellgren JH, Lawrence JS. Radiological assessment of osteo-arthrosis. *Annals of the Rheumatic Diseases*.
2. Huang G, Liu Z, Van Der Maaten L, Weinberger KQ. Densely Connected Convolutional Networks.
3. Xie S, Girshick R, Dollar P, Tu Z, He K. Aggregated Residual Transformations for Deep Neural Networks.
4. Hu J, Shen L, Sun G. Squeeze-and-Excitation Networks.
5. Selvaraju RR et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.
6. Osteoarthritis Initiative. Public knee radiograph cohort.
7. Knee Osteoarthritis Dataset with Severity, public Kaggle dataset used for the classification benchmark.

The repository contains the implementation, tests, experiment summaries, and report figures used to support this document.
