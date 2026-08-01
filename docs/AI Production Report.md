# HEALTHSYNC: DENSENET-121 PRODUCTION REPORT

Document status: production-only configuration  
Model family: DenseNet-121  
Checkpoint timestamp: `2026-07-30 09:03:29 UTC`

# CHAPTER 1: SYSTEM SCOPE

HealthSync is a computer-aided knee osteoarthritis grading prototype. It accepts a knee radiograph, detects each knee joint, predicts one Kellgren-Lawrence (KL) grade from 0 to 4 per detected knee, returns five class probabilities, and generates a Grad-CAM heatmap.

The system uses radiographs only. It does not use age, BMI, pain score, symptoms, or other clinical metadata. One input may contain one knee or a bilateral pair; detection and grading are performed per ROI. The returned KL class is an image-model estimate, not a clinical diagnosis. It must be interpreted with the source radiograph and patient context.

# CHAPTER 2: PRODUCTION PIPELINE

## 2.1. Knee Detection and ROI Construction

The production YOLOv8 checkpoint is:

`checkpoints/yolov8/2026-07-26_20-49-25_joint_detection/best.pt`

Each detected box is converted to the DenseNet input ROI as follows:

1. Calculate `crop_side = 1.15 * max(box_width, box_height)`.
2. Keep the square centered on the YOLO detection.
3. Clip the square to the source radiograph.
4. Add black padding only where the square extends outside the source image.
5. Preserve the full expanded ROI; do not center-crop it.

This policy preserves marginal osteophytes while limiting irrelevant femoral and tibial shaft content. It supports bilateral and single-knee radiographs. If YOLO detects no knee, the API returns an explicit no-ROI error instead of running the classifier on the full image.

## 2.2. Deterministic Classifier Preprocessing

The same preprocessing is used for validation, testing, and inference:

1. Convert the ROI to RGB.
2. Convert RGB to LAB and apply CLAHE to the lightness channel with `clipLimit=1.25` and `tileGridSize=(8, 8)`.
3. Square-pad any residual non-square ROI with black pixels.
4. Resize directly to `384 x 384`.
5. Convert to a floating-point tensor.
6. Normalize using ImageNet statistics:
   - mean: `[0.485, 0.456, 0.406]`
   - standard deviation: `[0.229, 0.224, 0.225]`

Natural left/right orientation is retained. Production inference does not use laterality canonicalization, horizontal flipping, test-time augmentation, Otsu thresholding, gamma correction, Gaussian noise, or random erasing.

## 2.3. Input and Failure Handling

The detector defines the classifier's field of view. The classifier never falls back to grading the full radiograph when no joint is detected. A no-detection condition returns an explicit API error so that detector failure cannot silently become a KL prediction. The square operation expands the shorter box dimension rather than stretching anatomy, and padding is introduced only when the expanded square lies outside the original image.

The preprocessing contract is checkpoint-specific. Changing ROI expansion, CLAHE strength/order, resize resolution, normalization, or laterality handling creates a different input distribution and requires a controlled evaluation before deployment.

# CHAPTER 3: PRODUCTION DENSENET-121

## 3.1. Architecture

The classifier is the standard ImageNet-initialized `timm` DenseNet-121:

| Component | Production configuration |
| --- | --- |
| Backbone | `densenet121` |
| Dense-block layers | `6, 12, 24, 16` |
| Global pooling | Global average pooling |
| Feature dimension | `1024` |
| Classifier | Linear `1024 -> 5` |
| Classifier dropout | `0.20` |
| Output | Five KL logits, followed by softmax for probabilities |
| Architecture identifier | `timm_densenet121_linear_gradcam` |

The production model does not use a hidden `1024 -> 256 -> 5` classifier or a native class-map head.

## 3.2. Loss and Class Balancing

All production training stages use five-class cross-entropy (CE). MSE, CORAL, CORN, focal CORN, ordinal PD-2, label smoothing, and hybrid ordinal losses are not used by the deployed checkpoint.

The training loader uses a full inverse-frequency `WeightedRandomSampler`:

- sampler power: `1.0`
- replacement: enabled
- sampled epoch length: equal to the training-set length
- minority-only augmentation: disabled

## 3.3. Training Augmentation

Only the training loader uses stochastic augmentation:

| Operation | Setting |
| --- | --- |
| Horizontal flip | probability `0.50` |
| Rotation | `+/-5` degrees |
| Brightness jitter | `0.08` |
| Contrast jitter | `0.08` |
| Random erasing | probability `0.10` |
| Erasing scale | `0.02-0.05` |
| Erasing ratio | `0.5-2.0` |

Gamma correction, Gaussian noise, double cutout, minority-only augmentation, and EMA are disabled.

## 3.4. Base Three-Stage Training

Common settings are batch size `48`, seed `42`, CUDA automatic mixed precision when available, AdamW, and global gradient-norm clipping at `1.0`. Early stopping and EMA are disabled.

| Stage | Epochs | Trainable parameters | Learning rate | Weight decay | Scheduler |
| --- | ---: | --- | --- | ---: | --- |
| Head warm-up | 5 | linear classifier | head `3e-4` | `1e-4` | none |
| Coarse fine-tuning | 15 | dense blocks 3-4, `norm5`, classifier | backbone `3e-5`; head `3e-4` | `1e-4` | cosine to `1e-7` |
| Full fine-tuning | 10 | complete network, restarted from best coarse weights | `1e-5` | `1e-3` | cosine to `1e-7` |

Checkpoints are selected using validation only:

`selection = 0.55 * QWK + 0.30 * macro_F1 + 0.15 * macro_AP`

## 3.5. Paired-View Production Adaptation

The selected base checkpoint was fine-tuned for five additional full-network epochs. Each training item used either the published knee crop or its production-style YOLO square ROI with probability `0.50` for each view.

| Parameter | Value |
| --- | --- |
| Loss | CE |
| Epochs | 5 |
| Batch size | 48 |
| Optimizer | AdamW |
| Learning rate | `1e-5` |
| Weight decay | `1e-3` |
| Scheduler | cosine annealing to `1e-7` |
| Gradient clipping | `1.0` |
| ROI expansion | `1.15` |
| Selected epoch | 4 |

Validation evaluates published crops and fixed production YOLO ROIs separately. The robust selection score is the mean of both domains' `0.55 QWK + 0.30 macro F1 + 0.15 macro AP` scores.

## 3.6. Reproducibility Record

The base and adaptation notebooks retain the executed outputs used to select this artifact. The checkpoint is identified by both its timestamped directory and SHA-256 digest; a generic filename alone is not sufficient provenance. Training seed `42` controls Python, NumPy, PyTorch, sampler, and loader generators. CUDA kernels and package versions can still introduce small numerical variation, so a retrained artifact must receive a new timestamp and evaluation record even when its configuration is unchanged.

# CHAPTER 4: PRODUCTION EVALUATION

## 4.1. Deployed Artifact

| Item | Value |
| --- | --- |
| Model mode | `densenet121` |
| Checkpoint | `checkpoints/densenet121/2026-07-30_09-03-29_850983_UTC_paired_view_yolo_roi/best_model.pth` |
| SHA-256 | `c9561cb4a76b64b11b5f4848036e3553f65aae3cc310099dbe638077c92578ca` |
| Architecture | `timm_densenet121_linear_gradcam` |
| Loss | CE |
| Paired-view probability | `0.50` |
| ROI expansion | `1.15` |
| Selected adaptation epoch | 4 |

## 4.2. Locked Production-ROI Test Result

The deployed checkpoint was evaluated on `1,656` labeled production-style YOLO square ROIs with the deterministic production transform.

| Metric | Result |
| --- | ---: |
| Accuracy | `0.5972` |
| QWK | `0.7702` |
| Macro precision | `0.6177` |
| Macro recall | `0.6420` |
| Macro F1 | `0.6215` |
| Macro AP | `0.6696` |
| Macro ROC AUC (OvR) | `0.8611` |

| KL grade | Support | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 639 | 0.6967 | 0.7371 | 0.7163 |
| 1 | 296 | 0.2832 | 0.3750 | 0.3227 |
| 2 | 447 | 0.6588 | 0.4362 | 0.5249 |
| 3 | 223 | 0.7269 | 0.7399 | 0.7333 |
| 4 | 51 | 0.7231 | 0.9216 | 0.8103 |

The model has strong Grade 4 recall but weak Grade 1 discrimination. No confidence intervals were exported for this exact evaluation, so none are claimed.

## 4.3. Production Grad-CAM

The API generates predicted-class Grad-CAM from `backbone.features.norm5`, the final normalized convolutional features before global pooling and classification:

1. Run a differentiable forward pass.
2. Select the predicted class logit.
3. Backpropagate the logit to `features.norm5`.
4. Spatially average the gradients to obtain feature-channel weights.
5. Calculate the weighted feature sum and apply ReLU.
6. Resize and normalize the map to the exact DenseNet ROI.
7. Overlay the map on that ROI.

No native CAM is used. The following evaluation panels show one successful and one failed case for every true KL grade. Each panel contains the original ROI, the predicted-class Grad-CAM, and the true-class Grad-CAM. The failed panels are deliberately retained: a heatmap can overlap the joint and still support the wrong grade.

| True grade | Successful prediction | Failed prediction |
| ---: | --- | --- |
| 0 | ![Grade 0 successful Grad-CAM](report/dense_net_121/runs/2026-07-30_09-03-29_paired_view_yolo_roi/assets/gradcam_success_grade_0.png) | ![Grade 0 failed Grad-CAM](report/dense_net_121/runs/2026-07-30_09-03-29_paired_view_yolo_roi/assets/gradcam_failure_grade_0.png) |
| 1 | ![Grade 1 successful Grad-CAM](report/dense_net_121/runs/2026-07-30_09-03-29_paired_view_yolo_roi/assets/gradcam_success_grade_1.png) | ![Grade 1 failed Grad-CAM](report/dense_net_121/runs/2026-07-30_09-03-29_paired_view_yolo_roi/assets/gradcam_failure_grade_1.png) |
| 2 | ![Grade 2 successful Grad-CAM](report/dense_net_121/runs/2026-07-30_09-03-29_paired_view_yolo_roi/assets/gradcam_success_grade_2.png) | ![Grade 2 failed Grad-CAM](report/dense_net_121/runs/2026-07-30_09-03-29_paired_view_yolo_roi/assets/gradcam_failure_grade_2.png) |
| 3 | ![Grade 3 successful Grad-CAM](report/dense_net_121/runs/2026-07-30_09-03-29_paired_view_yolo_roi/assets/gradcam_success_grade_3.png) | ![Grade 3 failed Grad-CAM](report/dense_net_121/runs/2026-07-30_09-03-29_paired_view_yolo_roi/assets/gradcam_failure_grade_3.png) |
| 4 | ![Grade 4 successful Grad-CAM](report/dense_net_121/runs/2026-07-30_09-03-29_paired_view_yolo_roi/assets/gradcam_success_grade_4.png) | ![Grade 4 failed Grad-CAM](report/dense_net_121/runs/2026-07-30_09-03-29_paired_view_yolo_roi/assets/gradcam_failure_grade_4.png) |

Grad-CAM indicates pixels that influenced a class score; it is not an osteophyte or joint-space-narrowing segmentation. Independent normalization also makes heatmap intensity incomparable between cases. The visual review should therefore check ROI adequacy, joint-line coverage, border/shaft shortcuts, prediction correctness, and predicted-versus-true class evidence together.

## 4.4. Limitations

- Grade 1 is the main classification weakness: precision `0.2832`, recall `0.3750`, and F1 `0.3227`.
- Grade 2 recall is `0.4362`, showing substantial adjacent-grade confusion even though its precision is higher.
- Grade 4 recall is high, but the class has only `51` test samples; this estimate is less stable than Grade 0.
- Classification depends on YOLO crop geometry and source-image domain. Published-crop metrics are not a substitute for production-ROI metrics.
- Grad-CAM is post-hoc and has no expert osteophyte or joint-space-narrowing masks for lesion-level validation.
- The exact production-ROI evaluation does not include confidence intervals, and the test set has been inspected across prior development iterations.
- The system has not undergone external prospective clinical validation and is not suitable for autonomous diagnosis.

# CHAPTER 5: DEPLOYMENT

The Python FastAPI service executes this flow:

`radiograph -> YOLOv8 -> 1.15 square ROI -> CLAHE 1.25 -> resize 384 -> DenseNet-121 -> probabilities -> Grad-CAM`

Model checkpoints are mounted read-only into the Docker container. `MODEL_MODE=densenet121` must be set, and the configured checkpoint architecture must match `timm_densenet121_linear_gradcam`. Startup validation should fail on a missing checkpoint or incompatible metadata. The API response schema remains stable while internal model implementation details change.

Operational verification covers health readiness, no-ROI behavior, single- and bilateral-knee inputs, probability normalization, valid KL range, ROI/heatmap geometry, and preservation of the public response schema. A successful health check proves service availability only; it does not prove clinical accuracy.

## Production Evidence

| Purpose | Evidence |
| --- | --- |
| Production base training | [Executed base notebook](report/dense_net_121/runs/2026-07-30_07-08-32_original_224_ce_3stage/notebook.ipynb) |
| Paired-view adaptation | [Executed adaptation notebook](report/dense_net_121/runs/2026-07-30_09-03-29_paired_view_yolo_roi/train_notebook.ipynb) |
| Locked evaluation and Grad-CAM figures | [Executed evaluation notebook](report/dense_net_121/runs/2026-07-30_09-03-29_paired_view_yolo_roi/evaluation_notebook.ipynb) |
| Production checkpoint record | [DenseNet report](report/dense_net_121/report.md) |
| Full experimental justification | [Complete AI report](AI%20Report.md) |

## Conclusion

This DenseNet-121 configuration uses a linear five-class head, CE loss, mild augmentation, inverse-frequency sampling, three-stage base training, and five-epoch paired-view adaptation. Inference uses natural laterality, production YOLO square crops, LAB CLAHE `1.25`, `384 x 384` input, and predicted-class Grad-CAM.
