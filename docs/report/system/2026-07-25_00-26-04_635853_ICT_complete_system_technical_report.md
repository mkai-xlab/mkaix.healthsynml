# Knee Osteoarthritis KL-Grading System: Complete Technical Report

**Report timestamp:** 2026-07-25 00:26:04.635853 ICT  
**Equivalent UTC timestamp:** 2026-07-24 17:26:04.635269 UTC  
**System state documented:** deployed two-classifier production pipeline tested on 2026-07-24  
**Primary task:** Kellgren-Lawrence (KL) Grade 0-4 classification from knee radiographs  
**Production classifiers:** DenseNet-121 and SE-ResNeXt50-32x4d  
**ROI detector:** YOLOv8n  
**Explanation method:** native class activation map (native CAM) with per-case anatomy gating  
**Deployment mode:** two-model weighted probability soft voting  

## 1. Report Scope and Evidence

This report describes the system that is actually implemented and deployed. It is
based on the executed training notebooks, saved checkpoint metadata, application
source code, CAM comparison, model reports, ensemble reports, and the complete
105-image remote API test. Values labeled as validation or test metrics come from
the corresponding saved experiment; they were not inferred from deployment images.

The deployed API test images have no KL ground-truth labels. That test establishes
operational correctness, response-contract stability, image decoding, probability
normalization, heatmap generation, and latency. It does not measure clinical
accuracy, QWK, precision, recall, F1, AP, AUC, sensitivity, or specificity.

## 2. Executive Summary

The production system is a sequential detection, classification, explanation, and
rendering pipeline:

```text
Uploaded radiograph
  -> decode image
  -> YOLOv8n knee ROI detection
  -> determine anatomical side
  -> mirror anatomical right knees into canonical orientation
  -> square padding + CLAHE + resize 400 + center crop 384
  -> DenseNet-121 native-CAM classifier
  -> SE-ResNeXt50-32x4d native-CAM classifier
  -> probability soft vote: 0.55 DenseNet + 0.45 SE-ResNeXt
  -> choose final KL grade by argmax
  -> evaluate both predicted-grade CAMs with an anatomy gate
  -> render the best acceptable native CAM
  -> return the established JSON response
```

The active classifiers share the same data split, deterministic evaluation
preprocessing, CE loss, full inverse-frequency sampler, validation selection
objective, and three-stage transfer-learning schedule. This makes their standalone
results meaningfully comparable.

DenseNet-121 is the stronger general classifier: test accuracy `0.6612`, macro F1
`0.6811`, Grade 1 recall `0.4493`, AP `0.7334`, and AUC `0.8987`. SE-ResNeXt has a
negligibly higher QWK (`0.8194` versus `0.8178`) and substantially cleaner broad-ROI
CAM statistics. Their QWK confidence intervals overlap, so the difference is not a
demonstrated superiority result. The models are retained together because their
strengths are complementary.

EfficientNet-B0 was evaluated as a third classifier but is not loaded in production.
It was weaker than both active models and no labeled paired validation demonstrated
that it improved the ensemble. Excluding it reduces latency and prevents a weaker
model from changing the vote or supplying poor per-case heatmaps.

Native CAM is used instead of Grad-CAM for operational reasons, not because it is
more anatomically accurate. With the implemented `1x1 convolution -> global average`
head, final-layer Grad-CAM and native CAM are analytically and empirically almost
identical. Native CAM needs only the forward pass and directly exposes the spatial
map averaged into the class logit.

The system is appropriate for a controlled research demonstration. It is not yet
validated as an autonomous clinical diagnostic product. Its largest remaining
limitations are adjacent-grade confusion, low confidence on many deployment cases,
coarse `12x12` localization, lack of patient-grouped external validation, CPU
latency, and lack of expert lesion/compartment annotations.

## 3. Clinical Prediction Target

The classifiers output one of five KL grades:

| Grade | API label | System description |
| ---: | --- | --- |
| 0 | `0Normal` | No radiographic signs of osteoarthritis |
| 1 | `1Doubtful` | Doubtful joint-space narrowing and possible osteophytic lipping |
| 2 | `2Mild` | Definite osteophytes and possible joint-space narrowing |
| 3 | `3Moderate` | Multiple osteophytes, definite narrowing, and some sclerosis |
| 4 | `4Severe` | Large osteophytes, marked narrowing, severe sclerosis, and deformity |

KL grading is ordinal, but the retained experiments use five-class cross-entropy.
CORN and other ordinal objectives were tested in earlier DenseNet experiments. An
older focal-CORN run reported higher QWK than the production CE checkpoint, so this
report does not claim that CE is the best historical metric result. CE was retained
because it supports the controlled five-map native-CAM architecture, canonical
laterality, strict checkpoint provenance, and the current metric/localization
comparison. Every grade logit is the spatial average of its own map.

## 4. Data and Split Protocol

### 4.1 Classification dataset

The classifiers use the Kaggle Knee Osteoarthritis Dataset with Severity. The
executed notebooks read the supplied folder splits and remove exact duplicate image
content by SHA-256-like file digest logic before accepting an image into a later
split.

| Split | Unique images | Use |
| --- | ---: | --- |
| Train | 5,778 | Parameter optimization |
| Validation | 826 | Epoch and checkpoint selection |
| Test | 1,656 | Final evaluation after checkpoint selection |
| Total | 8,260 | All retained unique images |

Training distribution:

| KL grade | Training images | Share |
| ---: | ---: | ---: |
| 0 | 2,286 | 39.6% |
| 1 | 1,046 | 18.1% |
| 2 | 1,516 | 26.2% |
| 3 | 757 | 13.1% |
| 4 | 173 | 3.0% |

Test distribution:

| KL grade | Test images |
| ---: | ---: |
| 0 | 639 |
| 1 | 296 |
| 2 | 447 |
| 3 | 223 |
| 4 | 51 |

The severe imbalance, especially the small Grade 4 class, motivated sampling rather
than loss reweighting in the final models.

### 4.2 Leakage controls and remaining split limitation

For each split, duplicate content is removed within that split. Validation excludes
hashes already present in training. Test excludes hashes already present in either
training or validation. This prevents exact duplicate files from crossing splits.

The final notebooks do not prove patient-grouped separation. Exact-image
deduplication is weaker than patient-level grouping because two different knees,
visits, crops, or acquisitions from the same person can remain correlated. The
current metrics must therefore be described as image-level results. A future
external evaluation should group by patient ID if such identifiers become
available.

### 4.3 ROI detector dataset

YOLOv8n was fine-tuned on an independently annotated knee ROI detection dataset.
The training notebook describes approximately 800 annotated radiographs. Its saved
validation output contains 315 images and 357 knee instances. The detector has one
class, `knee`.

## 5. Shared Classifier Preprocessing

Training and application inference intentionally use matching deterministic spatial
and intensity operations. The order matters because a heatmap is valid only when it
is overlaid on the exact image geometry given to the classifier.

### 5.1 Common deterministic operations

1. Decode the PNG or JPEG into an RGB image.
2. Determine anatomical laterality from the ROI position when possible.
3. Mirror anatomical right knees horizontally into the same canonical orientation
   as left knees.
4. Pad the shorter image dimension with black pixels to create a square without
   changing aspect ratio.
5. Apply CLAHE to the LAB lightness channel with clip limit `2.0` and tile grid
   `8x8`.
6. Resize to `400x400`.
7. Use a `384x384` crop.
8. Convert to a float tensor.
9. Normalize with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation
   `[0.229, 0.224, 0.225]`.

Square padding avoids geometrically stretching a rectangular YOLO crop. CLAHE
increases local radiographic contrast. Canonical laterality removes an avoidable
left/right orientation difference and makes learned medial/lateral evidence more
consistent. Random horizontal flip is deliberately absent because unconstrained
flips would conflict with this anatomical convention.

### 5.2 Training-only augmentation

| Transform | Setting |
| --- | --- |
| Random rotation | `+/-5 degrees` |
| Brightness jitter | `0.08` |
| Contrast jitter | `0.08` |
| Crop | random `384x384` crop after resizing to `400x400` |
| Random erasing | probability `0.10`, scale `0.02-0.05`, ratio `0.5-2.0`, value `0` |
| Horizontal flip | disabled |
| Minority-only augmentation | disabled |

These are mild augmentations. They introduce plausible acquisition variation while
avoiding large rotations, aggressive erasing, or repeated minority transforms that
previous experiments associated with instability and Grade 0/1 confusion.

### 5.3 Validation, test, and API inference

Validation, test, and production use no stochastic augmentation and no test-time
augmentation. The `400x400` resized image is center-cropped to `384x384`. The
application retains this processed RGB image for CAM overlay, so the displayed image
and activation map have the same coordinates.

## 6. Shared Native-CAM Classification Head

Each classifier backbone returns its final convolutional feature tensor
`F in R^(C x H x W)`. A learned `1x1` convolution maps the channels to five class
maps:

```text
M_k(x, y) = sum_c W[k, c] * F_c(x, y) + b_k
```

The logit for grade `k` is the global spatial mean:

```text
z_k = mean over x,y of M_k(x, y)
```

The class probability is:

```text
p_k = exp(z_k) / sum_j exp(z_j)
```

For a `384x384` input, the final class maps are `12x12`; the spatial downsampling
factor is 32. The native CAM for target class `k` is:

```text
CAM_k = normalize(bilinear_upsample(ReLU(M_k), 384x384))
```

The ReLU retains positive evidence for the target class. Per-map maximum
normalization makes visualization consistent, but means color intensity is relative
within a case and must not be compared as an absolute disease score across patients.

The head design has two important properties:

- The classification logit and explanation share the same class map. The heatmap is
  therefore faithful to the implemented head.
- The original `12x12` resolution is coarse. Upsampling produces a smooth `384x384`
  image but cannot create anatomical detail absent from the source map.

## 7. Loss, Sampling, and Checkpoint Selection

### 7.1 Cross-entropy loss

For true class `y`, training minimizes:

```text
CE = -log(p_y)
```

No focal factor, ordinal threshold head, label smoothing, joint-guidance penalty, or
CAM localization penalty is active in the production checkpoints.

### 7.1.1 Alternative loss and guidance evidence

Historical DenseNet results used different heads and protocols, so they are useful
context but not direct controlled competitors to the production native-CAM run:

| Historical run | Accuracy | QWK | AP | AUC | Important qualification |
| --- | ---: | ---: | ---: | ---: | --- |
| Unregularized CE baseline | 0.6691 | 0.8058 | 0.7009 | 0.8798 | Older classifier/Grad-CAM protocol |
| Balanced CE | 0.6594 | 0.8283 | 0.7287 | 0.8993 | Aggressive balancing and older explanation protocol |
| Focal CORN peak | **0.6733** | **0.8394** | **0.7439** | **0.9073** | Higher historical metrics, but no directly equivalent five-grade native-CAM head |
| Production native-CAM CE | 0.6612 | 0.8178 | 0.7334 | 0.8987 | Canonical, strictly identified, and explanation-faithful production architecture |

The repeatedly evaluated test split and protocol differences prevent using this
table to claim that one loss universally wins. A future CE-versus-CORN comparison
must use the same patient-grouped splits, augmentation, sampler, backbone state,
selection rule, and locked final holdout. CORN also produces conditional ordinal
threshold outputs rather than five independent grade logits, so a faithful
grade-specific explanation head must be designed and validated rather than attached
afterward.

Weak joint guidance was also tested by fine-tuning the DenseNet native-CAM model
against a hand-defined broad joint band. Guidance weight `0.05` increased validation
QWK from `0.8054` to `0.8082` and peak-inside-joint from `0.9648` to `0.9780`, but
macro F1 fell from `0.6927` to `0.6832` and Grade 1 recall fell from `0.3987` to
`0.3856`. Because the rectangle can reward a lateral hotspot without identifying
JSN or an osteophyte, that checkpoint was correctly rejected.

### 7.2 Full inverse-frequency sampler

Every training image with label `y` receives sampling weight:

```text
w_i = 1 / count(y_i)
```

`WeightedRandomSampler` samples `5,778` items per epoch with replacement. In
expectation, each class contributes equally even though the source distribution is
imbalanced. The sampler is deterministic from seed `42` at initialization.

This improves representation of Grades 1, 3, and 4, but oversamples repeated
minority examples and can contribute to boundary confusion. Controlled sampler
ablations did not pass the predefined promotion gates, so full inverse sampling was
retained for the final comparable models.

### 7.3 Validation composite

Checkpoints are selected on validation data using:

```text
selection = 0.40 * QWK
          + 0.20 * macro_F1
          + 0.10 * macro_recall
          + 0.10 * Grade_1_recall
          + 0.15 * macro_AP
          + 0.05 * macro_AUC
```

This prevents QWK alone from hiding poor minority-class behavior. QWK receives the
largest weight because KL grades are ordered; macro F1 and macro recall keep all
grades relevant; explicit Grade 1 recall addresses the hardest boundary class; AP
and AUC use the full probability vector rather than only the argmax.

The test set is evaluated only after validation selects the checkpoint within each
final notebook. However, the same test split has been used across many historical
experiments, so it is now indirectly part of the development loop. A newly locked
holdout is still required for an unbiased final claim.

## 8. Shared Three-Stage Transfer Learning

Both active classifiers start from ImageNet-pretrained backbone weights and use
mixed-precision training when CUDA is available. Batch size is `48`, with four data
workers, pinned memory, and persistent workers.

### Stage 1: head warm-up

| Item | Value |
| --- | --- |
| Epochs | 5 |
| Trainable layers | native-CAM `1x1` class head only |
| Optimizer | AdamW |
| Learning rate | `3e-4` |
| Weight decay | `1e-4` |
| Scheduler | none |

The pretrained backbone is frozen so the randomly initialized five-map head first
learns a usable mapping without destabilizing all feature layers.

### Stage 2: coarse adaptation

| Item | Value |
| --- | --- |
| Epochs | 15 |
| Trainable layers | final backbone stages plus class head |
| Optimizer | AdamW |
| Backbone learning rate | `3e-5` |
| Head learning rate | `3e-4` |
| Weight decay | `1e-4` |
| Scheduler | cosine annealing, `T_max=15`, `eta_min=1e-7` |

DenseNet unfreezes parameters matching `denseblock3` and `denseblock4`.
SE-ResNeXt unfreezes final stages matching `layer3`, `layer4`, `stages.2`, or
`stages.3`. The best Stage 2 checkpoint is saved independently.

### Stage 3: full fine-tuning

| Item | Value |
| --- | --- |
| Epochs | 10 |
| Initialization | best Stage 2 checkpoint |
| Trainable layers | entire network |
| Optimizer | AdamW |
| Learning rate | `1e-5` |
| Weight decay | `1e-3` |
| Scheduler | cosine annealing, `T_max=10`, `eta_min=1e-7` |

The lower learning rate protects pretrained features while allowing global
adaptation. Every update uses automatic mixed precision on CUDA and gradient norm
clipping at `1.0`. DenseNet and SE-ResNeXt final checkpoint selection is restricted
to the fine-tuning stage and uses the validation composite above.

## 9. DenseNet-121 Production Classifier

### 9.1 Architecture

DenseNet-121 uses dense connectivity: each layer receives feature maps from earlier
layers in the same dense block. This supports feature reuse and strong gradient flow
with fewer parameters than many similarly deep networks.

Implementation details:

| Field | Value |
| --- | --- |
| Backbone | `timm densenet121`, ImageNet pretrained during training |
| Backbone mode | `features_only=True`, final output index only |
| Final feature channels | 1,024 |
| Native-CAM head | `Conv2d(1024, 5, kernel_size=1)` |
| Source class-map size | `12x12` |
| Checkpoint architecture | `canonical_final_linear_cam` |
| State-dict tensor elements | 7,042,750 including stored buffers |

### 9.2 Exact checkpoint

| Field | Value |
| --- | --- |
| Run timestamp | `2026-07-23 01:31:37.184239 UTC` |
| Run directory | `2026-07-23_01-31-37_184239_UTC_canonical_final_linear_cam_production` |
| Selected epoch | 27, fine-tuning stage |
| Validation selection score | 0.7276 |
| Local production file | `checkpoints/densenet121/best_model.pth` |
| File size | 28,469,032 bytes |
| SHA-256 | `cce1602b382411ada19883b180be501f333a5301de2c69aa00d61b031905efd1` |

### 9.3 Selected validation result

| Metric | Value |
| --- | ---: |
| Accuracy | 0.6683 |
| QWK | 0.8139 |
| Macro precision | 0.6901 |
| Macro recall | 0.7018 |
| Macro F1 | 0.6952 |
| Grade 1 recall | 0.4052 |
| Macro AP | 0.7198 |
| Macro AUC | 0.8877 |
| Validation loss | 0.8196 |
| Composite selection | 0.7276 |

### 9.4 Final test result

| Metric | Value |
| --- | ---: |
| Accuracy | 0.6612, image-bootstrap 95% CI 0.6383-0.6848 |
| QWK | 0.8178, image-bootstrap 95% CI 0.7971-0.8366 |
| Macro precision | 0.6873 |
| Macro recall | 0.6783 |
| Macro F1 | 0.6811 |
| Grade 1 recall | 0.4493 |
| Macro AP | 0.7334 |
| Macro AUC | 0.8987 |

Confusion matrix, with rows as true grades and columns as predictions:

| True grade | Pred 0 | Pred 1 | Pred 2 | Pred 3 | Pred 4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 476 | 131 | 31 | 1 | 0 |
| 1 | 100 | 133 | 57 | 6 | 0 |
| 2 | 46 | 94 | 268 | 39 | 0 |
| 3 | 0 | 6 | 33 | 177 | 7 |
| 4 | 0 | 0 | 2 | 8 | 41 |

Of 561 errors, 469 (`83.6%`) were adjacent-grade errors. The largest weakness is
reciprocal Grade 0/1 confusion, which is expected to be difficult because Grade 1
findings are doubtful by definition.

### 9.5 DenseNet CAM audit

The stratified validation audit used 227 cases, up to 50 per grade:

| CAM metric | Value |
| --- | ---: |
| Mean joint-ROI energy | 0.7996 |
| Mean border energy | 0.1323 |
| Mean lower-tibia energy | 0.1006 |
| Peak inside broad joint ROI | 1.0000 |

DenseNet provides the best overall classification balance, but its worst CAMs show
lateral-margin activation. A broad joint band counts these peaks as inside the ROI,
so the quantitative score must be interpreted together with the image gallery.

## 10. SE-ResNeXt50-32x4d Production Classifier

### 10.1 Architecture

SE-ResNeXt combines grouped residual transformations with squeeze-and-excitation
channel attention. The ResNeXt cardinality is 32 groups with width 4. The SE blocks
learn channel-wise recalibration before residual aggregation.

| Field | Value |
| --- | --- |
| Backbone | `timm seresnext50_32x4d`, ImageNet pretrained during training |
| Backbone mode | `features_only=True`, final output index only |
| Final feature channels | 2,048 |
| Native-CAM head | `Conv2d(2048, 5, kernel_size=1)` |
| Source class-map size | `12x12` |
| Checkpoint architecture | `final_native_cam_ce` |
| State-dict tensor elements | 25,589,418 including stored buffers |

### 10.2 Exact checkpoint

| Field | Value |
| --- | --- |
| Run timestamp | `2026-07-23 01:25:36.772175 UTC` |
| Run directory | `2026-07-23_01-25-36_772175_UTC_final_native_cam_ce` |
| Selected epoch | 24, fine-tuning stage |
| Validation selection score | 0.7003 |
| Local production file | `checkpoints/se_resnext50_32x4d/best_model (1).pth` |
| File size | 102,502,033 bytes |
| SHA-256 | `98630fdfe4618a11bdc0149b1ba7429c22ae2d63e5d3729a784957f30815af95` |

### 10.3 Selected validation result

| Metric | Value |
| --- | ---: |
| Accuracy | 0.6235 |
| QWK | 0.7873 |
| Macro precision | 0.6430 |
| Macro recall | 0.6606 |
| Macro F1 | 0.6498 |
| Grade 1 recall | 0.4183 |
| Macro AP | 0.6915 |
| Macro AUC | 0.8752 |
| Validation loss | 0.9028 |
| Composite selection | 0.7003 |

### 10.4 Final test result

| Metric | Value |
| --- | ---: |
| Accuracy | 0.6389, image-bootstrap 95% CI 0.6153-0.6624 |
| QWK | 0.8194, image-bootstrap 95% CI 0.7999-0.8384 |
| Macro precision | 0.6677 |
| Macro recall | 0.6727 |
| Macro F1 | 0.6671 |
| Grade 1 recall | 0.4155 |
| Macro AP | 0.7248 |
| Macro AUC | 0.8948 |
| Test loss | 0.7905 |

Confusion matrix:

| True grade | Pred 0 | Pred 1 | Pred 2 | Pred 3 | Pred 4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 437 | 170 | 29 | 3 | 0 |
| 1 | 99 | 123 | 71 | 3 | 0 |
| 2 | 34 | 96 | 269 | 48 | 0 |
| 3 | 0 | 5 | 22 | 187 | 9 |
| 4 | 0 | 0 | 0 | 9 | 42 |

### 10.5 SE-ResNeXt CAM audit

| CAM metric | Value |
| --- | ---: |
| Mean joint-ROI energy | 0.8707 |
| Mean border energy | 0.0749 |
| Mean lower-tibia energy | 0.0880 |
| Peak inside broad joint ROI | 0.9956 |

SE-ResNeXt has the strongest broad joint concentration among the three completed
classifier candidates. It still exhibits occasional lateral hotspots, so it should
not be described as a lesion segmenter.

### 10.6 Rejected SE-ResNeXt upgrade

The later `2026-07-23 06:57:13.378879 UTC` experiment fused 24x24 and 12x12 class
maps and used EMA decay `0.999`. Grade 1 recall rose to `0.5507`, but accuracy fell to
`0.5676`, QWK to `0.7651`, macro F1 to `0.6220`, AP to `0.6918`, and AUC to `0.8703`.
CAM joint energy also fell while border/lower-tibia energy increased. It failed the
promotion criteria and is correctly excluded from the application.

## 11. EfficientNet-B0 Candidate: Evaluated but Excluded

EfficientNet-B0 was selected from the completed B0-B3 scale comparison because it
had the best combined validation objective and was the smallest candidate. B4 did
not complete, so it cannot be compared fairly.

| Field | Value |
| --- | --- |
| Architecture | EfficientNet-B0 with final five-map native-CAM head |
| Final channels | 1,280 |
| Input/map size | `384x384` / `12x12` |
| Training | same 5 + 15 + 10 stage schedule |
| Physical/effective batch | 24 / 48 via two-step gradient accumulation |
| Selected epoch | 10, coarse stage |
| Test accuracy | 0.6051 |
| Test QWK | 0.7992 |
| Test macro F1 | 0.6258 |
| Test Grade 1 recall | 0.3986 |
| Test AP / AUC | 0.6817 / 0.8723 |
| Joint/border/lower-tibia energy | 0.8280 / 0.1080 / 0.0797 |

Checkpoint provenance:

| Field | Value |
| --- | --- |
| Run timestamp | `2026-07-24 04:45:25.604705 UTC` |
| Local file | `checkpoints/efficientnet_b0/best_model.pth` |
| File size | 16,358,137 bytes |
| SHA-256 | `47238a3ee5350b6521e3f292d30493e7a7e37d0c7aee748b46424c0859fe60ff` |

B0 was below DenseNet by `0.0561` accuracy, `0.0186` QWK, `0.0553` macro F1, and
`0.0517` AP. It was also below SE-ResNeXt by `0.0202` QWK and `0.0413` macro F1.
An unlabeled 20-image trial could not establish that adding B0 improved accuracy.
Production therefore assigns it weight `0.00` and does not load it in ensemble mode.
It remains available only as an explicit standalone comparison mode.

## 12. YOLOv8n Knee ROI Detector

### 12.1 Purpose

The detector finds one or both knee joints in a full radiograph and passes cropped
ROIs to the classifiers. This reduces irrelevant anatomy and background and makes
the classifier input distribution closer to the cropped training data.

### 12.2 Training configuration

| Parameter | Value |
| --- | --- |
| Initial model | COCO-pretrained `yolov8n.pt` |
| Task | single-class object detection |
| Image size | 640 |
| Batch size | 16 |
| Configured/completed epochs | 50 / 50 |
| Patience | 10 |
| Seed | 42 |
| Optimizer | Ultralytics `auto` |
| Initial LR | 0.001 |
| Final LR fraction | 0.01 |
| Weight decay | 0.0005 |
| Hue/saturation/value augmentation | 0.015 / 0.7 / 0.4 |
| Rotation | `+/-10 degrees` |
| Translation | 0.1 |
| Scale | 0.5 |
| Horizontal flip | 0.5 |
| Training GPU | Tesla T4 |

### 12.3 Validation result

The best saved detector was evaluated on 315 images containing 357 instances:

| Metric | Value |
| --- | ---: |
| Box precision | 0.989 |
| Box recall | 0.980 |
| mAP50 | 0.9880 |
| mAP75 | 0.8810 |
| mAP50-95 | 0.7456 |

The notebook reports a fused model with 73 layers, 3,005,843 parameters, and 8.1
GFLOPs. The validation log also warns that 13 segmentation records were present
alongside 357 boxes; Ultralytics discarded segments and evaluated detection boxes.
This should be cleaned in a future detector dataset revision even though detection
training completed.

### 12.4 Production detector behavior

The application uses detector confidence threshold `0.45`. Detections are sorted by
horizontal coordinate.

- With exactly two detections, the left image position is labeled anatomical
  `right`, and the right image position is labeled anatomical `left`.
- With one detection, a center left of 40% image width is called `right`; a center
  right of 60% is called `left`; otherwise laterality is `unknown`.
- With more than two detections, laterality is `unknown`.
- If no ROI is returned, the full image is classified with unknown laterality.

The deployed local file is `checkpoints/yolov8/best.pt`, size 24,496,935 bytes,
SHA-256
`3e18a09e58df0ae1f8e3102d5e63893c72d2508bba5ffd80c84dac8356161d2b`.
The application requires this mounted checkpoint for normal ROI behavior.

There is an unresolved provenance discrepancy: the executed YOLO notebook reports
that its stripped `best.pt` was 6.2 MB, whereas the checkpoint currently in the
repository is 24.5 MB. The available environment could not inspect the Ultralytics
checkpoint metadata without the Ultralytics package. Therefore, the training
configuration and metrics above are authoritative for the executed notebook, but
the current file's exact identity as that notebook's selected artifact has not been
cryptographically established. This does not invalidate the successful API smoke
test, but it must be resolved before claiming exact detector reproducibility.

## 13. Production Ensemble

### 13.1 Probability-level weighted soft voting

Each classifier produces five logits. The application applies softmax separately,
then computes:

```text
p_ensemble = 0.55 * p_DenseNet + 0.45 * p_SE-ResNeXt
predicted_grade = argmax(p_ensemble)
confidence = max(p_ensemble)
```

Weights are validated as finite and non-negative and normalized by their sum. The
application combines probabilities, not logits and not hard class labels.

The `0.55/0.45` choice favors DenseNet because it leads on accuracy, macro F1,
Grade 1 recall, AP, and AUC. SE-ResNeXt retains substantial weight because it has
similar ordinal performance and stronger broad-ROI CAM concentration. These weights
are a conservative engineering choice, not a learned optimum: a labeled paired
probability table and locked holdout are still needed to optimize and validate them.

### 13.2 Standalone classifier comparison

| Model | Accuracy | QWK | Macro F1 | G1 recall | AP | AUC | Joint energy | Border energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DenseNet-121 | **0.6612** | 0.8178 | **0.6811** | **0.4493** | **0.7334** | **0.8987** | 0.7996 | 0.1323 |
| SE-ResNeXt50 | 0.6389 | **0.8194** | 0.6671 | 0.4155 | 0.7248 | 0.8948 | **0.8707** | **0.0749** |
| EfficientNet-B0 | 0.6051 | 0.7992 | 0.6258 | 0.3986 | 0.6817 | 0.8723 | 0.8280 | 0.1080 |

No labeled test metric for the final two-model ensemble has been exported in the
available artifacts. Standalone test scores must not be presented as ensemble test
scores.

## 14. Heatmap Method

### 14.1 Why native CAM is used

The controlled validation audit compared native CAM and final-layer Grad-CAM on 227
images for DenseNet and SE-ResNeXt. Map correlation was `1.0000` for both models.
Mean absolute differences were `0.00014` for DenseNet and `0.00008` for SE-ResNeXt.
The predefined result was no demonstrated superiority for either method.

The comparison notebook's checkpoint resolver used substring matching and selected
`stage2_best_model.pth` rather than the intended exact `best_model.pth`. Therefore,
its exact aggregate localization values should not be presented as final-checkpoint
measurements. This resolver issue does not change the method conclusion: with this
linear final map and global-average head, the analytical equivalence still holds,
and the observed implementation errors were effectively zero.

For a linear class-map head followed by global average pooling, Grad-CAM weights at
that final map reduce to the same class-specific linear evidence up to scaling and
bias handling. Consequently, switching to Grad-CAM would not repair lateral-margin
hotspots. Native CAM is retained because it:

- needs one forward pass and no backward pass;
- avoids gradient hooks and `retain_grad` errors;
- uses less inference memory;
- is directly tied to the map averaged into the selected logit;
- preserves the existing API field while simplifying production.

The existing JSON key is still named `gradcam_image` for client compatibility, but
its content is a native-CAM overlay.

### 14.2 Per-case anatomy measurements

For each active model's CAM of the final ensemble-predicted grade, the application
defines normalized regions on the `384x384` map:

| Region | Normalized bounds |
| --- | --- |
| Broad joint ROI | y `0.28-0.72`, x `0.06-0.94` |
| Border | outside central y/x `0.08-0.92` rectangle |
| Lower tibia | y `0.72-0.96`, x `0.06-0.94` |

It computes:

```text
joint_energy       = CAM energy inside broad joint ROI / total CAM energy
border_energy      = CAM energy in image border / total CAM energy
lower_tibia_energy = CAM energy in lower-tibia band / total CAM energy
anatomy_score      = joint_energy * (1-border_energy) * (1-lower_tibia_energy)
```

It also records whether the maximum CAM pixel lies inside the broad joint ROI.

### 14.3 Anatomy gate and model selection

A component CAM passes when all conditions are true:

```text
joint_energy >= 0.55
border_energy <= 0.25
lower_tibia_energy <= 0.25
peak_inside_joint == true
```

Among passing component maps, the chosen source maximizes:

```text
anatomy_score * component_probability_for_the_ensemble_grade
```

The component argmax does not have to equal the ensemble argmax. This is intentional:
agreement alone previously forced visibly poor maps. If neither component passes,
the same score chooses the best available map and the server emits a warning.

### 14.4 Rendering and orientation

The selected normalized CAM is converted with OpenCV JET colors and blended as 60%
processed radiograph plus 40% heatmap. It is encoded as a JPEG data URL.

For an anatomical right knee, both the classifier input and the background used for
the CAM overlay remain in canonical mirrored orientation. The current pipeline
computes `was_mirrored` but does not use it to flip the overlay back. Consequently,
the returned `gradcam_image` is internally aligned with the model input, but it may
be horizontally reversed relative to the separately returned raw `roi_image`. This
does not change the prediction or CAM/model alignment, but it can confuse a user
comparing the two images side by side. The application should either flip the final
overlay back before returning it or label the overlay explicitly as canonical
orientation. This is a presentation/alignment correction, not a model architecture
change.

### 14.5 What the heatmap proves and does not prove

The map faithfully shows positive spatial evidence used by the final classifier
head for one grade. It does not prove that a hotspot is an osteophyte, joint-space
narrowing, sclerosis, or exact lesion boundary. The anatomy gate is a quality filter
based on broad rectangular regions, not learned anatomical segmentation.

Therefore, correct reporting language is: "the model's positive evidence is mainly
at joint level." Incorrect language is: "the heatmap precisely localizes the OA
lesion." Exact claims require expert compartment landmarks, JSN/osteophyte labels,
or segmentation masks and a separate held-out localization evaluation.

## 15. End-to-End API Execution

For `POST /api/v1/predict`:

1. FastAPI reads the uploaded file bytes.
2. OpenCV validates and decodes the image.
3. YOLOv8 detects knee boxes at confidence `0.45`.
4. The service determines side and crops each ROI.
5. Every ROI is preprocessed independently.
6. Both classifiers run once and return logits plus native class maps.
7. Soft voting creates the final five-grade probability vector.
8. The top probability becomes the grade and confidence.
9. Both component CAMs are extracted for that same final grade.
10. The anatomy gate chooses one CAM source.
11. The overlay, ROI, bounding box, detector confidence, and laterality are attached.
12. The original image is annotated with box, grade, side, and confidence.
13. Pydantic validates the response.

Established response shape:

```json
{
  "filename": "image.png",
  "predictions": [
    {
      "predicted_class": 0,
      "predicted_grade": "0Normal",
      "confidence": 0.0,
      "description": "...",
      "details": {
        "0Normal": 0.0,
        "1Doubtful": 0.0,
        "2Mild": 0.0,
        "3Moderate": 0.0,
        "4Severe": 0.0
      },
      "box": [0, 0, 0, 0],
      "yolo_confidence": 0.0,
      "knee_side": "right",
      "roi_image": "data:image/png;base64,...",
      "gradcam_image": "data:image/jpeg;base64,..."
    }
  ],
  "annotated_image": "data:image/jpeg;base64,..."
}
```

The service also exposes `/api/v1/predict/detect-roi`, `/api/v1/health`, and
`/api/v1/models`. Checkpoint architecture and model-name metadata are checked at
startup and state dicts load strictly. Missing or incompatible required classifier
checkpoints stop startup rather than falling back to random weights.

## 16. Deployed API Verification

The public deployment at `http://54.254.113.71:8005` was tested from
`2026-07-24 16:50:26.365170 UTC` through
`2026-07-24 17:15:24.638256 UTC`.

| Check | Result |
| --- | --- |
| Source images attempted | 105 |
| HTTP 200 responses | 105/105 |
| Knee predictions | 209 |
| Sources with two knees | 104 |
| Sources with one knee | 1 |
| Empty prediction lists | 0 |
| Probability vectors summing to one | 209/209 |
| Decodable CAM overlays | 209/209 at `384x384` |
| Decodable ROI and annotated images | all |
| Request timeouts | 0 |
| HTTP 4xx/5xx | 0 |
| Response schema | unchanged |

Prediction distribution on the unlabeled deployment images:

| Grade | Count | Share |
| ---: | ---: | ---: |
| 0 | 132 | 63.2% |
| 1 | 36 | 17.2% |
| 2 | 26 | 12.4% |
| 3 | 13 | 6.2% |
| 4 | 2 | 1.0% |

Confidence and latency:

| Measure | Value |
| --- | ---: |
| Mean confidence | 0.4298 |
| Median confidence | 0.3870 |
| Predictions below 0.40 | 110/209, 52.6% |
| Mean response time | 11.893 s |
| Median response time | 10.253 s |
| p95 response time | 25.553 s |
| Maximum response time | 38.501 s |

The run demonstrates stable behavior but shows two operational problems: more than
half of predictions have low maximum probability, and CPU latency is too variable
for a consistently responsive interactive workflow.

Visual review found that anatomy gating corrected the worst diffuse upper-femur and
lower-tibia maps from the previous selector. Some maps still emphasize one lateral
joint margin or retain secondary edge activation. This is improved broad joint
localization, not perfect radiographic feature localization.

## 17. Known Limitations and Risks

### 17.1 Predictive limitations

- Grade 0/1 and Grade 1/2 confusion remain the dominant errors.
- Grade 4 has only 173 training and 51 test examples, so its estimates are uncertain.
- Full balancing repeatedly samples minority images and can increase overfitting.
- The reported bootstrap intervals are image-level, not patient-clustered.
- The repeatedly used test set is no longer a pristine final holdout.
- The final ensemble itself lacks a saved labeled paired evaluation.
- Deployment probabilities are not proven calibrated.

### 17.2 Localization limitations

- `12x12` source maps are spatially coarse.
- Bilinear interpolation cannot recover missing anatomical resolution.
- The broad anatomy gate can accept a lateral-margin hotspot.
- Native CAM identifies model evidence, not pathology boundaries.
- There are no expert JSN, osteophyte, compartment, or landmark annotations.
- CAM quality metrics and classifier metrics can move in opposite directions.

### 17.3 Pipeline limitations

- Laterality inference assumes conventional bilateral radiograph arrangement.
- A single centrally positioned knee is assigned unknown laterality.
- Right-knee CAM overlays are returned in canonical mirrored orientation while the
  raw ROI image is returned in its original orientation.
- If YOLO misses all knees, the full image is classified, which is distributionally
  different from a normal ROI input.
- The current YOLO checkpoint's file size does not match the stripped artifact size
  recorded by the training notebook, so detector checkpoint provenance is incomplete.
- The endpoint does not enforce a documented upload-size cap in the inspected code.
- The public deployment uses plain HTTP and publicly exposed documentation.
- CPU inference and base64 image serialization produce high and variable latency.

## 18. Recommended Actions

### Immediate application improvements; no architecture change required

1. Present a non-diagnostic low-confidence message when confidence is below `0.40`,
   while preserving the API schema unless the client contract is deliberately
   versioned.
2. Add stage timing for decode, YOLO, preprocessing, DenseNet, SE-ResNeXt, CAM, JPEG
   encoding, and response serialization.
3. Record internal heatmap source, gate pass/fallback status, joint energy, border
   energy, lower-tibia energy, and predicted probabilities in structured server
   logs, not in the existing client response.
4. Flip canonical right-knee overlays back to original ROI orientation before
   returning them, or make the canonical orientation explicit in the client.
5. Identify the exact YOLO checkpoint run, archive its executed notebook and
   metrics with the checkpoint, and record its SHA-256 in one manifest.
6. Add HTTPS, access control where appropriate, rate limiting, request-size limits,
   restricted security-group rules, and removal/restriction of public API docs.
7. Pin the Python, PyTorch, timm, torchvision, OpenCV, Ultralytics, and CUDA/runtime
   versions used for production.

### Evaluation improvements before retraining

1. Export DenseNet and SE-ResNeXt probabilities for the exact same labeled cases.
2. Fit temperature scaling on validation data before comparing ensemble weights.
3. Search the two-model weight interval using the predefined QWK/F1/Grade 1
   objective and calibration as a secondary metric.
4. Freeze weights, then evaluate exactly once on a newly locked patient-level
   holdout.
5. Bootstrap confidence intervals by patient ID when identifiers are available.
6. Audit at least 50 cases per grade with blinded expert review in addition to
   rectangular energy metrics and occlusion tests.

### Architecture or training changes only after evidence justifies them

Do not change the current architecture merely because a small number of heatmaps are
visually imperfect. Grad-CAM will not fix the learned evidence pattern. If the new
locked evaluation confirms a repeatable problem, the most defensible next training
experiment is expert-supervised localization or auxiliary radiographic-feature
prediction using future JSN/osteophyte labels. Without those annotations, another
broad CAM penalty risks rewarding the same lateral hotspots.

## 19. Current Production Decision

Retain the current architecture and checkpoints:

- YOLOv8n for knee ROI extraction;
- DenseNet-121 epoch 27 as the higher-weight classifier;
- SE-ResNeXt50-32x4d epoch 24 as the complementary classifier and stronger
  localization candidate;
- probability soft voting at `0.55/0.45`;
- native CAM with the per-case anatomy gate;
- EfficientNet-B0 excluded from the production ensemble;
- unchanged prediction JSON schema.

The next work should focus on uncertainty presentation, latency profiling,
deployment security, logging, and a newly locked labeled evaluation. There is not
currently enough evidence to justify replacing the model architecture.

## 20. Reproducibility and Artifact Index

| Artifact | Location |
| --- | --- |
| DenseNet production notebook | [`notebooks/production/dense_net_121_production.ipynb`](../../../notebooks/production/dense_net_121_production.ipynb) |
| DenseNet complete run report | [`docs/report/dense_net_121/report.md`](../dense_net_121/report.md) |
| SE-ResNeXt executed notebook | [`docs/report/se_resnext50_32x4d/2026-07-23_01-25-36_seresnext50_32x4d_final_native_cam_ce.ipynb`](../se_resnext50_32x4d/2026-07-23_01-25-36_seresnext50_32x4d_final_native_cam_ce.ipynb) |
| SE-ResNeXt complete run report | [`docs/report/se_resnext50_32x4d/report.md`](../se_resnext50_32x4d/report.md) |
| EfficientNet-B0 executed notebook | [`docs/report/efficientnet_b0/2026-07-24_04-45-25_efficientnet_b0_final_native_cam_ce.ipynb`](../efficientnet_b0/2026-07-24_04-45-25_efficientnet_b0_final_native_cam_ce.ipynb) |
| EfficientNet report | [`docs/report/efficientnet_b0/report.md`](../efficientnet_b0/report.md) |
| YOLO training notebook | [`notebooks/yolov8_knee_detection.ipynb`](../../../notebooks/yolov8_knee_detection.ipynb) |
| Grad-CAM/native-CAM comparison | [`docs/report/cam_comparison/report.md`](../cam_comparison/report.md) |
| Complete remote API test | [`docs/report/ensemble/2026-07-24_16-50-26_365170_UTC_deployed_api_full_test.md`](../ensemble/2026-07-24_16-50-26_365170_UTC_deployed_api_full_test.md) |
| Remote heatmap montage | [`docs/report/ensemble/assets/2026-07-24_16-50-26_365170_UTC_remote_heatmap_montage.jpg`](../ensemble/assets/2026-07-24_16-50-26_365170_UTC_remote_heatmap_montage.jpg) |
| Production pipeline code | [`app/ml/pipelines/knee_oa_pipeline.py`](../../../app/ml/pipelines/knee_oa_pipeline.py) |
| Preprocessing code | [`app/services/preprocessing_service.py`](../../../app/services/preprocessing_service.py) |
| Native-CAM code | [`app/services/gradcam_service.py`](../../../app/services/gradcam_service.py) |
| ROI service | [`app/services/roi_service.py`](../../../app/services/roi_service.py) |

## 21. Method References

1. Kellgren, J. H. and Lawrence, J. S. Radiological assessment of osteo-arthrosis.
   *Annals of the Rheumatic Diseases*, 1957.
   [DOI](https://doi.org/10.1136/ard.16.4.494)
2. Huang, G. et al. Densely Connected Convolutional Networks. CVPR 2017.
   [Paper](https://arxiv.org/abs/1608.06993)
3. Xie, S. et al. Aggregated Residual Transformations for Deep Neural Networks.
   CVPR 2017. [Paper](https://arxiv.org/abs/1611.05431)
4. Hu, J. et al. Squeeze-and-Excitation Networks. CVPR 2018.
   [Paper](https://arxiv.org/abs/1709.01507)
5. Zhou, B. et al. Learning Deep Features for Discriminative Localization. CVPR
   2016. [DOI](https://doi.org/10.1109/CVPR.2016.319)
6. Selvaraju, R. R. et al. Grad-CAM: Visual Explanations from Deep Networks via
   Gradient-Based Localization. ICCV 2017.
   [DOI](https://doi.org/10.1109/ICCV.2017.74)
7. Tan, M. and Le, Q. V. EfficientNet: Rethinking Model Scaling for Convolutional
   Neural Networks. ICML 2019. [Paper](https://arxiv.org/abs/1905.11946)
8. Adebayo, J. et al. Sanity Checks for Saliency Maps. NeurIPS 2018.
   [Paper](https://arxiv.org/abs/1810.03292)
