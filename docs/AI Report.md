# HEALTHSYNC: DENSENET-121 FOR KELLGREN-LAWRENCE GRADING

Document status: DenseNet-121 production report  
Production checkpoint: `2026-07-30_09-03-29_850983_UTC_paired_view_yolo_roi`  
Scope: the deployed DenseNet-121 classifier and the experiments that directly support its configuration

# CHAPTER 1: INTRODUCTION

## 1.1. Problem Statement

Knee osteoarthritis (KOA) severity is commonly summarized with the Kellgren-Lawrence (KL) scale, an ordinal scale from Grade 0 (no radiographic OA) to Grade 4 (severe OA). Manual grading is difficult because joint-space narrowing, osteophytes, sclerosis, and deformity vary gradually, especially near the Grade 0/1 and Grade 1/2 boundaries.

HealthSync is a computer-aided grading prototype. It detects each knee joint in an uploaded radiograph, predicts one KL grade per detected knee, returns the five class probabilities, and generates a Grad-CAM heatmap. The model output is supportive evidence only; it is not a diagnosis and does not replace review by a qualified clinician.

## 1.2. Objectives

The DenseNet-121 work has four objectives:

1. Detect knee-joint regions from full radiographs with YOLOv8 and preserve marginal anatomy in a reproducible square crop.
2. Fine-tune an ImageNet-initialized DenseNet-121 for five-class KL grading.
3. Select training, loss, sampling, preprocessing, and ROI policies using validation evidence rather than repeatedly tuning against the test set.
4. expose predicted grade, probabilities, and post-hoc Grad-CAM through the FastAPI inference service.

This version intentionally excludes SE-ResNeXt and EfficientNet. Their experiments belong in their own reports and must not be mixed with the production-only DenseNet-121 result in Chapter 4.

## 1.3. Scope

The system uses plain knee radiographs without age, BMI, pain score, or other clinical metadata. It estimates the current KL grade and does not predict progression. Its explanation is a class-discriminative heatmap, not a segmentation of joint-space narrowing or osteophytes.

# CHAPTER 2: BACKGROUND

## 2.1. Kellgren-Lawrence Grades

| Grade | Radiographic interpretation |
| ---: | --- |
| 0 | No radiographic evidence of osteoarthritis. |
| 1 | Doubtful joint-space narrowing and possible osteophytic lipping. |
| 2 | Definite osteophytes with possible joint-space narrowing. |
| 3 | Multiple osteophytes, definite narrowing, possible sclerosis, and possible deformity. |
| 4 | Large osteophytes, marked narrowing, severe sclerosis, and definite deformity. |

The classes are ordered, but adjacent grades overlap visually. Therefore, the report gives QWK and macro metrics alongside accuracy and reports each grade separately.

## 2.2. DenseNet-121

DenseNet connects each layer to all subsequent layers in the same dense block. DenseNet-121 uses four dense blocks with 6, 12, 24, and 16 layers. Feature reuse and direct gradient paths make it a practical transfer-learning backbone for a relatively small medical-image dataset.

The implemented model is the standard `timm` DenseNet-121 with ImageNet initialization, global average pooling, dropout `0.20`, and one linear `1024 -> 5` classifier. It is **not** the earlier experimental five-map native-CAM head and does not use a hidden `1024 -> 256 -> 5` head.

## 2.3. Explainability Boundary

Grad-CAM weights final convolutional feature maps by the gradient of the selected class logit. It shows which spatial regions most influenced that class score. It does not prove that the model found a clinically valid lesion. For that reason, classification performance and heatmap localization are assessed independently.

# CHAPTER 3: METHODOLOGY AND MODEL SELECTION

## 3.1. Production Input Pipeline

The production inference path is fixed as follows:

1. YOLOv8 detects each knee joint using `checkpoints/yolov8/2026-07-26_20-49-25_joint_detection/best.pt`.
2. For each detection, the crop side is `1.15 * max(box_width, box_height)` and remains centered on the detected box.
3. The resulting square is clipped to the source radiograph. Black padding is added only where the square extends outside the source image.
4. LAB-space CLAHE is applied to the lightness channel with `clipLimit=1.25` and an `8 x 8` tile grid.
5. Any residual non-square image is square-padded; no center crop is used.
6. The ROI is resized directly to `384 x 384`, converted to three-channel RGB tensor form, and normalized with ImageNet mean and standard deviation.
7. Left and right knees retain their natural orientation. Inference does not use flipping, test-time augmentation, Otsu thresholding, gamma correction, Gaussian noise, or laterality canonicalization.

The `1.15` expansion preserves femoral and tibial margins, where osteophytes may appear, while limiting irrelevant shaft and border content. This exact crop policy is important: a classifier trained only on the published 224-pixel crops did not transfer reliably to differently framed YOLO crops.

## 3.2. Training Data and Augmentation

The base classifier uses the five-grade `kneeKL224` train and validation directories from KneeXrayData. The production adaptation also uses paired YOLO-derived square ROIs generated from the corresponding full radiographs. A training item selects the published crop or the production-style ROI with equal probability (`0.50`). Validation evaluates the two views separately.

Training-only augmentation is deliberately mild:

| Operation | Setting |
| --- | --- |
| Horizontal flip | probability `0.50` |
| Rotation | up to `+/-5` degrees |
| Brightness jitter | `0.08` |
| Contrast jitter | `0.08` |
| Random erasing | probability `0.10`, scale `0.02-0.05`, ratio `0.5-2.0` |
| Gamma / Gaussian noise | disabled |
| Minority-only augmentation | disabled |
| Test-time augmentation | disabled |

The experiment history includes stronger erasing, gamma, noise, canonical orientation, and alternative CLAHE orderings. These are documented as experiments, but they are not production transforms.

## 3.3. Training Schedule

### 3.3.1. Base Three-Stage Training

The base checkpoint was trained with batch size `48`, seed `42`, mixed precision, gradient-norm clipping at `1.0`, and AdamW. Full inverse-frequency `WeightedRandomSampler` sampling (`power=1.0`, replacement enabled) was used for training. EMA and early stopping were disabled.

| Stage | Epochs | Trainable parameters | Learning rate | Weight decay | Scheduler |
| --- | ---: | --- | --- | ---: | --- |
| Head warm-up | 5 | linear classifier | head `3e-4` | `1e-4` | none |
| Coarse fine-tuning | 15 | dense blocks 3-4, `norm5`, classifier | backbone `3e-5`; head `3e-4` | `1e-4` | cosine, `eta_min=1e-7` |
| Full fine-tuning | 10 | complete network, restarted from best coarse checkpoint | `1e-5` | `1e-3` | cosine, `eta_min=1e-7` |

At every validation epoch, the checkpoint objective was:

`selection = 0.55 * QWK + 0.30 * macro_F1 + 0.15 * macro_AP`

This avoids selecting on QWK alone while keeping ordinal agreement as the primary term.

### 3.3.2. Paired-View Production Adaptation

The selected base checkpoint was then fine-tuned for five full-network epochs with the 50/50 published/YOLO view policy. This stage used CE, AdamW with learning rate `1e-5`, weight decay `1e-3`, cosine annealing to `1e-7`, batch size `48`, mixed precision, gradient clipping `1.0`, and the same inverse-frequency sampler.

Each epoch was scored on published validation crops and fixed production-style YOLO validation ROIs. The robust score was the mean of both domains' selection scores. Epoch 4 produced the deployed checkpoint. MSE was not used in either the base training or paired-view adaptation.

Evidence: [base training notebook](report/dense_net_121/runs/2026-07-30_07-08-32_original_224_ce_3stage/notebook.ipynb) and [paired-view adaptation notebook](report/dense_net_121/runs/2026-07-30_09-03-29_paired_view_yolo_roi/train_notebook.ipynb).

## 3.4. Loss Functions: Evaluated, Historical, and Unused

Techniques that affected model selection should remain in the report even when they are not deployed. Removing them would hide negative results and make the CE decision impossible to audit. They must, however, be clearly separated from production.

| Loss | DenseNet-121 status | Evidence and decision |
| --- | --- | --- |
| Cross-entropy (CE) | **Production** | Won the controlled CE/PD-2 study and is used by the paired-view checkpoint. |
| Ordinal PD-2 | Controlled evaluation | High Grade 1 recall but severe collapse in macro F1 and AP; rejected. |
| CE + `0.25` ordinal PD-2 | Controlled evaluation | Close to CE but did not exceed the validation selector; rejected. |
| CORN | Historical evaluation | Completed run exists, but it used an older architecture/configuration and is not a controlled comparison with production. |
| Focal CORN | Historical evaluation | Several completed runs exist; one achieved strong historical test metrics, while two later variants contain documented unfreezing logic errors. Not valid evidence to replace the production loss. |
| CORAL | **Not evaluated as a production arm** | Legacy helper functions exist in the base notebook, but the active model only accepts CE/PD-2 variants and no completed CORAL result exists. CORAL must not be described as tested. |
| Ordinal soft-label loss | Not present in the final DenseNet study | No completed, attributable result was found; omitted from performance claims. |

### 3.4.1. Controlled CE vs Ordinal PD-2 Result

The run `2026-07-25_06-30-25_175448_UTC_final_noncanonical_loss_ablation` compared all three arms under the same split, seed, sampler, augmentation, architecture, and schedule. Selection used validation only.

| Loss arm | Best epoch | Accuracy | QWK | Macro F1 | Grade 1 recall | Macro AP | Selection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **CE** | **24** | **0.6465** | **0.8083** | **0.6819** | 0.3856 | **0.7310** | **0.7588** |
| Ordinal PD-2 | 29 | 0.3862 | 0.6679 | 0.3352 | **0.8824** | 0.3521 | 0.5207 |
| CE + `0.25` PD-2 | 30 | 0.6453 | 0.8066 | 0.6783 | 0.4314 | 0.7308 | 0.7567 |

The hybrid loss was competitive but did not beat CE. Pure PD-2 biased predictions toward Grade 1 and reduced overall discrimination. This controlled validation result, rather than the repeatedly inspected test set, is the primary evidence for CE.

Evidence: [executed loss-ablation notebook](report/dense_net_121/runs/2026-07-25_06-30-25_final_noncanonical_loss_ablation/executed_notebook.ipynb) and the named 2026-07-25 loss-ablation section in the [consolidated DenseNet report](report/dense_net_121/report.md).

### 3.4.2. CORN and Focal CORN Context

Historical experiments remain useful, but they do not isolate loss as the only variable. Plain CORN reached test QWK `0.8246` in the 2026-07-20 run. The best historical focal-CORN run reached accuracy `0.6733`, QWK `0.8394`, macro AP `0.7439`, and AUC `0.9073` on 2026-07-17. Those figures are higher than some CE runs, so it would be incorrect to write “CE always outperformed CORN.” They came from an older crop, head, schedule, and repeatedly inspected test workflow and therefore cannot justify replacing the production CE checkpoint.

Evidence: [plain CORN notebook](report/dense_net_121/runs/2026-07-20_12-36-36_corn/notebook.ipynb), [best historical focal-CORN notebook](report/dense_net_121/runs/2026-07-17_16-06-42_focal_corn_optimized_lr_patience/notebook.ipynb), and the [historical comparison table](report/dense_net_121/report.md#historical-single-configuration-runs).

## 3.5. Other Configuration Decisions

| Technique | Production decision | Supporting evidence |
| --- | --- | --- |
| Natural laterality + training flip | selected | Canonicalization is disabled; a flip is augmentation only, so single-knee images remain valid inputs. |
| CLAHE `1.25` before padding | selected | Validation preprocessing ablation selected this ordering; production applies the same deterministic transform. |
| Full inverse sampler | retained | Used by the selected base and adaptation runs. It is a production fact, not proof that it is universally optimal. |
| EMA | rejected | The completed natural-orientation gamma/EMA run collapsed to accuracy `0.4553` and QWK `0.7248`; it changed several variables, so the result rejects that package rather than EMA in isolation. |
| Strong/double cutout | rejected for production | Historical metric gains did not establish better localization; production uses only mild random erasing. |
| Native CAM head | replaced | Production uses the standard linear DenseNet classifier and Grad-CAM at `features.norm5`. |
| Joint-guided CAM loss | not promoted | A broad rectangular guidance proxy improved its own localization score but reduced macro F1 and Grade 1 recall. |
| ROI robustness fine-tune | not promoted | Improved classification metrics but worsened joint energy and border energy on the same CAM audit set. |

Preprocessing evidence: the [quality-ablation section](report/dense_net_121/report.md#preprocessing-comparison), [quality notebook](report/dense_net_121/runs/2026-07-25_23-48-22_preprocessing_quality_ablation/notebook.ipynb), and [confirmation notebook](report/dense_net_121/runs/2026-07-26_06-44-07_preprocessing_confirmation/notebook.ipynb). Heatmap and ROI evidence: [joint-guided notebook](report/dense_net_121/runs/2026-07-22_11-52-13_joint_guided_cam_ablation/notebook.ipynb) and [production ROI robustness notebook](report/dense_net_121/runs/2026-07-30_15-45-03_production_roi_robustness/notebook.ipynb).

## 3.6. Production Grad-CAM

The API computes predicted-class Grad-CAM from `backbone.features.norm5`, the final normalized convolutional feature tensor before global pooling and the linear classifier. The process is:

1. run a differentiable forward pass and select the predicted class logit;
2. backpropagate that logit to `features.norm5`;
3. spatially average each feature map's gradients to obtain channel weights;
4. compute the weighted feature sum and apply ReLU;
5. resize and normalize the map to the ROI dimensions; and
6. overlay the map on the exact ROI supplied to DenseNet-121.

The heatmap is generated from the class score, not from YOLO confidence. No native-CAM map is used in the production DenseNet path. During evaluation, misclassified cases also generate separate predicted-class and true-class CAMs to expose what supported the wrong decision.

## 3.7. Evaluation Criteria

The report uses accuracy, QWK, macro precision, macro recall, macro F1, macro one-vs-rest AP, macro one-vs-rest ROC AUC, per-grade precision/recall/F1, and the confusion matrix. QWK measures ordinal agreement; macro metrics prevent Grade 0 from dominating the summary.

Heatmap quality is assessed separately with joint-band energy, border energy, peak-inside-joint rate, occlusion response, and visual review of correctly and incorrectly classified cases. A high QWK does not imply an anatomically faithful CAM.

# CHAPTER 4: PRODUCTION DENSENET-121 EVALUATION

## 4.1. Deployed Artifact

| Item | Production value |
| --- | --- |
| Model mode | `densenet121` |
| Architecture | `timm_densenet121_linear_gradcam` |
| Loss | CE |
| Checkpoint | `checkpoints/densenet121/2026-07-30_09-03-29_850983_UTC_paired_view_yolo_roi/best_model.pth` |
| SHA-256 | `c9561cb4a76b64b11b5f4848036e3553f65aae3cc310099dbe638077c92578ca` |
| Selected adaptation epoch | 4 of 5 |
| Input | production YOLO square ROI, `384 x 384` |
| Explanation | predicted-class Grad-CAM from `features.norm5` |

## 4.2. Locked Production-ROI Test Result

The production checkpoint was evaluated once on `1,656` labeled YOLO-square ROIs using the same deterministic preprocessing as deployment. This is the result attributable to the deployed classifier; historical experiments are intentionally excluded from this chapter.

| Metric | Result |
| --- | ---: |
| Accuracy | `0.5972` |
| QWK | `0.7702` |
| Macro precision | `0.6177` |
| Macro recall | `0.6420` |
| Macro F1 | `0.6215` |
| Macro AP | `0.6696` |
| Macro ROC AUC (OvR) | `0.8611` |

| True grade | Support | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 639 | 0.6967 | 0.7371 | 0.7163 |
| 1 | 296 | 0.2832 | 0.3750 | 0.3227 |
| 2 | 447 | 0.6588 | 0.4362 | 0.5249 |
| 3 | 223 | 0.7269 | 0.7399 | 0.7333 |
| 4 | 51 | 0.7231 | 0.9216 | 0.8103 |

The strongest result is Grade 4 recall (`0.9216`). The principal weakness is Grade 1 discrimination: precision `0.2832`, recall `0.3750`, and F1 `0.3227`. Accuracy and macro F1 are therefore modest despite useful ordinal agreement. No confidence intervals were exported for this exact production-ROI evaluation, so none are claimed here.

Evidence: [production evaluation notebook](report/dense_net_121/runs/2026-07-30_09-03-29_paired_view_yolo_roi/evaluation_notebook.ipynb). Its executed output directory is `2026-07-30_11-53-13_842110_UTC`.

## 4.3. Production Grad-CAM Audit

The evaluation notebook exports correct examples grouped by true grade and predicted-vs-true CAM pairs for misclassified cases. A separate fixed-ROI validation audit evaluated `227` cases and recorded the following baseline values for the deployed checkpoint:

| Heatmap metric | Production baseline |
| --- | ---: |
| Mean joint-band energy | `0.8648` |
| Mean border energy | `0.0761` |
| Peak inside joint rate | `0.9956` |

A later five-epoch ROI robustness candidate improved validation accuracy (`0.5884 -> 0.6186`) and QWK (`0.7405 -> 0.7650`) but reduced joint energy (`0.8648 -> 0.8505`) and increased border energy (`0.0761 -> 0.0844`). It was correctly not promoted. This result demonstrates why Chapter 4 reports model performance and explanation quality separately.

The Grad-CAM result remains a localization aid, not lesion-level ground truth. A hotspot on a marginal osteophyte can be plausible even if it is lateral to the central joint space; activation on image padding, the upper femoral shaft, or lower tibial shaft is more suspicious. Definitive explanation validation would require expert region annotations or occlusion/segmentation targets.

## 4.4. Production Assessment

The artifact is suitable as a capstone prototype because the detector, classifier, checkpoint provenance, deterministic preprocessing, quantitative evaluation, and visual explanation audit are reproducible. It is not validated for autonomous clinical use. The main unresolved limitations are low Grade 1 performance, source/crop domain sensitivity, absence of expert lesion masks for explanation validation, and lack of confidence intervals for the exact production-ROI test result.

The next improvement should not be another uncontrolled loss change. It should use the exact production YOLO crop pipeline, keep a locked labeled holdout, compare one change at a time, and require both classification and CAM non-inferiority before promotion.

# CHAPTER 5: SOFTWARE ARCHITECTURE AND DEPLOYMENT

## 5.1. Inference Flow

The Python FastAPI service accepts an image, runs YOLOv8, creates one square ROI per detected knee, applies the production DenseNet preprocessing, predicts five-class probabilities, and generates Grad-CAM. If no joint is detected, the API returns an explicit no-ROI error response rather than fabricating a classifier result.

Checkpoints are mounted read-only into the Docker container. Environment configuration selects the model mode and exact YOLO and DenseNet paths. The API response schema is kept stable while model and heatmap internals evolve.

## 5.2. Evidence Index

| Purpose | Evidence |
| --- | --- |
| Complete DenseNet run history | [DenseNet consolidated report](report/dense_net_121/report.md) |
| Machine-readable run inventory | [DenseNet summary CSV](report/dense_net_121/experiment_summary.csv) |
| Production base training | [Executed base notebook](report/dense_net_121/runs/2026-07-30_07-08-32_original_224_ce_3stage/notebook.ipynb) |
| Paired-view production adaptation | [Executed adaptation notebook](report/dense_net_121/runs/2026-07-30_09-03-29_paired_view_yolo_roi/train_notebook.ipynb) |
| Locked production evaluation and CAM figures | [Executed evaluation notebook](report/dense_net_121/runs/2026-07-30_09-03-29_paired_view_yolo_roi/evaluation_notebook.ipynb) |
| Controlled CE/ordinal loss decision | [Loss ablation notebook](report/dense_net_121/runs/2026-07-25_06-30-25_final_noncanonical_loss_ablation/executed_notebook.ipynb) |
| Production ROI robustness rejection | [Robustness notebook](report/dense_net_121/runs/2026-07-30_15-45-03_production_roi_robustness/notebook.ipynb) |

## 5.3. Conclusion

The current HealthSync classifier is a standard DenseNet-121 trained with CE, mild augmentation, inverse-frequency sampling, three-stage fine-tuning, and a short paired published/production-ROI adaptation. It uses natural laterality and Grad-CAM, not canonicalization or native CAM. CE was selected by a controlled validation comparison; historical CORN results are retained as evidence but are not directly comparable to production. The locked production-ROI test confirms strong severe-grade recall and useful ordinal agreement, while also showing that Grade 1 classification and explanation validation remain open research limitations.

## References

- Huang, G. et al. “Densely Connected Convolutional Networks,” CVPR 2017.
- Selvaraju, R. R. et al. “Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,” ICCV 2017.
- Shi, X. et al. “Deep Neural Networks for Rank-Consistent Ordinal Regression Based on Conditional Probabilities,” Pattern Recognition 2023 (CORN).
- Cao, W. et al. “Rank Consistent Ordinal Regression for Neural Networks with Application to Age Estimation,” Pattern Recognition Letters 2020 (CORAL).
