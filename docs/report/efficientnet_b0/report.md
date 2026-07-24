# EfficientNet-B0 Training Execution Log
This file records the EfficientNet scale ablation that selected B0 and the completed standalone B0 evaluation against the current production candidates.

## Model Performance Summary

| Exact UTC Run Timestamp | Experiment | Data Used | Status | Current Decision |
| --- | --- | --- | --- | --- |
| 2026-07-24 04:45:25.604705 UTC | EfficientNet-B0 final 12x12 native-CAM CE | Train, validation, and test | Completed all 30 epochs, test evaluation, and 227-case CAM audit | Do not promote as a standalone production model; retain only as a paired ensemble candidate |
| 2026-07-23 16:08:50.642801 UTC | EfficientNet B0/B1/B2/B3/B4 controlled scale ablation | Train and validation only | Incomplete during B4 epoch 16; CAM audit and final comparison did not run | B0 is the best completed scale candidate, but it is not production-ready |

## Run: 2026-07-24 04:45:25.604705 UTC (EFFICIENTNET-B0 - FINAL 12X12 NATIVE CAM CE)

### Summary
The standalone B0 run completed all 30 epochs without a notebook error. Global checkpoint selection correctly retained coarse epoch 10 instead of replacing it with a weaker fine-tuning checkpoint. The selected model obtained test QWK `0.7992`, macro F1 `0.6258`, and macro AP `0.6817`. Its native CAM generally remains on the knee joint, but the gallery still contains strong medial or lateral boundary hotspots and predicted-class and true-class maps that are nearly identical for several misclassified cases.

The result confirms B0 as the appropriate EfficientNet scale for this dataset, but it does not beat either current production candidate. It should not replace DenseNet-121 or SE-ResNeXt-50 and should not be added automatically to the ensemble.

### Fixed Configuration

| Parameter | Value |
| --- | --- |
| Backbone | EfficientNet-B0, ImageNet pretrained |
| Input | Resize 400x400, crop 384x384 |
| Head | Final 1x1 convolution producing five 12x12 class maps; global means are CE logits |
| Loss | Cross-entropy |
| Sampler | Full inverse-frequency |
| Laterality | Right knees mirrored to the canonical orientation |
| Training | 5 warm-up + 15 coarse + 10 full-finetune epochs |
| Selected checkpoint | Epoch 10, coarse stage |
| Batch / accumulation / effective batch | 24 / 2 / 48 |
| EMA / multiscale fusion | Disabled / disabled |
| Train / validation / test size | 5,778 / 826 / 1,656 unique images |
| GPU | Tesla T4 |

### Selected Validation Metrics

| Metric | Value |
| --- | ---: |
| Accuracy | 0.6150 |
| QWK | 0.7743 |
| Macro precision | 0.6279 |
| Macro recall | 0.6616 |
| Macro F1 | 0.6317 |
| Grade 1 recall | 0.4771 |
| Macro AP | 0.6690 |
| Macro AUC | 0.8660 |
| Selection score | 0.6936 |

The strongest fine-tuning candidate, around epoch 29, had QWK `0.7798` and macro F1 `0.6342`, but Grade 1 recall fell to `0.2876`; its selection score was therefore only `0.6793`. Keeping epoch 10 is consistent with the predefined combined objective rather than QWK alone.

### Test Metrics

| Metric | Value |
| --- | ---: |
| Loss | 0.8843 |
| Accuracy | 0.6051 |
| QWK | 0.7992 |
| Macro precision | 0.6260 |
| Macro recall | 0.6418 |
| Macro F1 | 0.6258 |
| Grade 1 recall | 0.3986 |
| Macro AP | 0.6817 |
| Macro AUC | 0.8723 |

| Grade | Precision | Recall | F1 | Support |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.72 | 0.74 | 0.73 | 639 |
| 1 | 0.29 | 0.40 | 0.34 | 296 |
| 2 | 0.68 | 0.45 | 0.54 | 447 |
| 3 | 0.71 | 0.77 | 0.74 | 223 |
| 4 | 0.73 | 0.86 | 0.79 | 51 |

The confusion matrix rows are true grades and columns are predicted grades:

| True grade | Pred 0 | Pred 1 | Pred 2 | Pred 3 | Pred 4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 470 | 148 | 21 | 0 | 0 |
| 1 | 127 | 118 | 49 | 2 | 0 |
| 2 | 59 | 126 | 199 | 63 | 0 |
| 3 | 1 | 14 | 21 | 171 | 16 |
| 4 | 0 | 0 | 1 | 6 | 44 |

Grade 1 remains the main weakness: precision is only `0.29`, with reciprocal Grade 0/1 confusion. Grade 2 recall is also low at `0.45`. Grade 4 recall is strong, but its support is only 51 images.

### Native-CAM Audit

The audit evaluated 227 cases with 12x12 source class maps.

| CAM measure | Value | Interpretation |
| --- | ---: | --- |
| Joint-ROI energy | 0.8280 | Most activation is in the broad joint region |
| Border energy | 0.1080 | Better than DenseNet-121, but worse than SE-ResNeXt-50 |
| Lower-tibia energy | 0.0797 | Lowest of the three final candidates |
| Peak inside joint ROI | 0.9956 | Nearly every peak falls inside the broad ROI |
| CAM/occlusion correlation | 0.6172 | Moderate faithfulness agreement |
| Top-CAM occlusion drop | 0.0985 | Occluding the strongest CAM area usually lowers evidence |

Visual review is still required alongside these aggregate measures. Several worst-case examples place a concentrated hotspot at an outer joint margin instead of distributing attention across the tibiofemoral joint space. In several Grade 1 to Grade 0 errors, predicted- and true-class CAMs are almost indistinguishable. The CAM is broadly localized and reasonably faithful, but it is not anatomically perfect and does not resolve the model's adjacent-grade discrimination problem.

### Comparison with Current Final Models

| Model | Accuracy | QWK | Macro F1 | Grade 1 Recall | Macro AP | Macro AUC | Joint Energy | Border Energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DenseNet-121 | **0.6612** | 0.8178 | **0.6811** | **0.4493** | **0.7334** | **0.8987** | 0.7996 | 0.1323 |
| SE-ResNeXt-50 | 0.6389 | **0.8194** | 0.6671 | 0.4155 | 0.7248 | 0.8948 | **0.8707** | **0.0749** |
| EfficientNet-B0 | 0.6051 | 0.7992 | 0.6258 | 0.3986 | 0.6817 | 0.8723 | 0.8280 | 0.1080 |

Compared with DenseNet-121, B0 is lower by `0.0561` accuracy, `0.0186` QWK, `0.0553` macro F1, and `0.0517` macro AP. Compared with SE-ResNeXt-50, it is lower by `0.0202` QWK and `0.0413` macro F1. Its CAM localization is between the two models: broader joint localization is better than DenseNet-121, while SE-ResNeXt-50 has substantially higher joint energy and lower border energy.

### Decision and Next Step

**Decision:** do not promote EfficientNet-B0 as the standalone production model. Keep DenseNet-121 as the default predictive model and SE-ResNeXt-50 as the stronger localization-oriented secondary model. The B0 result does not justify more B1-B4 training.

EfficientNet-B0 may still provide complementary errors, so the only justified next experiment is an evaluation-only paired ensemble comparison using saved per-image probabilities from DenseNet-121, SE-ResNeXt-50, and B0 on exactly the same validation cases. Compare the current two-model ensemble with the three-model ensemble and retain B0 only if it improves QWK, macro F1, and calibration without using the repeatedly evaluated test set to select weights. A newly locked holdout remains necessary for the final claim.

### Archived Notebook

- [Executed EfficientNet-B0 final run](2026-07-24_04-45-25_efficientnet_b0_final_native_cam_ce.ipynb)

## Run: 2026-07-23 16:08:50.642801 UTC (EFFICIENTNET SCALE ABLATION - INCOMPLETE)

### Summary
The experiment trained B0, B1, B2, and B3 for all 30 epochs under the same split, transforms, full inverse-frequency sampler, CE loss, staged unfreezing schedule, and 12x12 native-CAM head. B4 reached coarse epoch 16 before the Colab server disconnected. The notebook therefore never ran its cross-scale CAM audit or final comparison table. No test data were read.

B0 produced the highest observed composite validation score among the completed scales (`0.6936`) and is much smaller than B3. This supports B0 as the only EfficientNet scale worth a standalone completion run. It does not establish that B0 is better than DenseNet-121 or SE-ResNeXt-50, and it does not establish acceptable CAM localization.

### Fixed Configuration

| Parameter | Value |
| --- | --- |
| Input | Resize 400x400, crop 384x384 |
| Head | Final 1x1 convolution producing five 12x12 class maps; global means are CE logits |
| Loss | Cross-entropy |
| Sampler | Full inverse-frequency |
| Laterality | Right knees mirrored to the canonical orientation |
| Training | 5 warm-up + 15 coarse + 10 full-finetune epochs |
| Effective Batch Size | 48 for every scale |
| EMA / multiscale fusion | Disabled / disabled |
| Train / validation size | 5,778 / 826 unique images |
| Test split | Not read |

### Completed Scale Results

The table reports the best observed validation composite across all completed epochs, including the coarse stage. The original ablation also saved a separate final-stage candidate, so the standalone B0 notebook now retains the best checkpoint across both coarse and fine-tuning stages.

| Scale | Best Epoch / Stage | Selection | QWK | Macro F1 | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| **B0** | **10 / coarse** | **0.6936** | 0.7743 | **0.6317** | Best completed combined objective and smallest model |
| B1 | 17 / coarse | 0.6811 | 0.7831 | 0.6266 | Lower combined objective than B0 |
| B2 | 14 / coarse | 0.6875 | 0.7680 | 0.6279 | Close, but no gain over B0 |
| B3 | 26 / finetune | 0.6830 | 0.7798 | 0.6270 | Peak QWK reached 0.7963 at epoch 18, but the combined objective remained lower |
| B4 | 14 / coarse, incomplete | 0.6591 | 0.7565 | 0.5947 | Training and CAM evaluation incomplete; not comparable |

The final-stage B0 candidate peaked at epoch 28 with QWK `0.7794`, macro F1 `0.6402`, and selection `0.6830`. The stronger epoch-10 composite is why checkpoint selection must span coarse and fine-tuning stages instead of discarding the coarse winner.

### CAM and Production Assessment
No EfficientNet CAM audit was produced by this run. The planned 50-per-grade joint energy, border energy, peak location, and occlusion evaluation never executed. The unrun B4 candidate notebook also contains no evidence and must not be cited as a completed experiment.

**Decision:** do not place EfficientNet-B0 or EfficientNet-B4 in production and do not include either in an ensemble yet. If a third-model comparison is required for the report, run [the fixed B0 notebook](../../../notebooks/efficientnet_b0.ipynb) once from top to bottom. Promote it only after its standalone validation, one locked evaluation, CAM audit, and paired ensemble comparison are complete.

### Research Context
EfficientNet proposes compound scaling of depth, width, and resolution, but it does not imply that a larger scale is optimal for a small medical dataset ([Tan and Le, 2019](https://arxiv.org/abs/1905.11946)). The current empirical result supports the smaller B0 scale. Earlier knee-OA work likewise shows that compact transfer-learning backbones can be competitive and that performance depends strongly on data size and validation design ([Antony et al., 2016](https://doi.org/10.1109/ICPR.2016.7899799); [Thomas et al., 2020](https://doi.org/10.1148/ryai.2020190065)).

### Archived Notebook

- [Executed scale ablation, interrupted during B4](2026-07-23_16-08-50_efficientnet_b_scale_ablation_incomplete.ipynb)
