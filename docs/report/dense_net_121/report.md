# DenseNet-121 Experiment Register

This is the single human-readable index for DenseNet-121. It records configuration and results only. Complete notebooks and figures are stored under [`../archive/dense_net_121/runs/`](../archive/dense_net_121/runs/); the single machine-readable index is [`experiment_summary.csv`](experiment_summary.csv). Metrics from validation, published-crop test, and production-YOLO-ROI test are labeled separately and must not be compared as if they used the same image domain.

## 384 x 384 Paired-View Production Run

This sequence uses a 224x224 base model, then resizes both published and YOLO ROI views to 384x384 during paired-view adaptation and evaluation. Therefore, Notebook 02 is the 224x224 base-training step; Notebooks 03 and 04 are the 384x384 production-domain steps.

| Notebook | Purpose | Input | Output |
| --- | --- | --- | --- |
| [02 original base training](../../../notebooks/densenet121/runs/2026-07-30_07-08-32_original_224_ce_3stage.ipynb) | Train the CE base model | 224x224 | Initialization checkpoint |
| `03 paired-view adaptation` | Fine-tune on published crops + YOLO ROIs | 384x384 | Paired-view checkpoint; notebook file is not retained |
| `04 locked Grad-CAM evaluation` | Evaluate the locked YOLO-ROI test set | 384x384 | Metrics and Grad-CAM galleries; notebook file is not retained |

The locked 384x384 production-ROI test result was Accuracy `0.5972`, QWK `0.7702`, macro F1 `0.6215`, AP `0.6696`, and AUC `0.8611`.

## 224 x 224 Paired-ROI Resolution Comparison

Run date: `2026-08-04` to `2026-08-05 UTC`
Archived dataset preparation: [01 dataset preparation](../../../notebooks/datasets/01_prepare_original_and_yolo_roi_datasets.ipynb). The 224x224 training, adaptation, and evaluation notebook sources are not retained; their recorded metrics are preserved below.

Notebook 01 did not need to rebuild the derived YOLO ROI images: Notebook 03 successfully consumed the existing `densenet121_yolo_square_roi_trainvaltest_v2` train/validation set. Notebook 02 trained the same five-logit CE DenseNet-121 configuration at `224 x 224`; the selected base checkpoint was from epoch `28` (validation selection `0.7193`). Its published-crop test result was Accuracy `0.6184`, QWK `0.7931`, AP `0.6796`, and AUC `0.8722`.

Notebook 03 fine-tuned this checkpoint for five epochs with the same `50/50` published-crop and YOLO-square-ROI mixture. Epoch `4` was selected by the validation-only mean-domain score: published QWK `0.7778`, YOLO-ROI QWK `0.7191`, and robust selection `0.6988`. Notebook 04 then evaluated that selected checkpoint once on the locked `n=1,656` YOLO-ROI test set and generated post-hoc Grad-CAM evidence.

| Input | Accuracy | QWK | Macro F1 | Macro AP | Macro AUC | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `224 x 224` paired view | 0.5743 | 0.7366 | 0.5906 | 0.6367 | 0.8479 | rejected |
| `384 x 384` paired view | **0.5972** | **0.7702** | **0.6215** | **0.6696** | **0.8611** | retained |

The two rows use the same locked production-style ROI test split, YOLO square-crop policy, CE loss, five-epoch paired-view adaptation, and predicted-class Grad-CAM procedure. `384 x 384` improved every reported locked-test metric: QWK by `0.0336`, macro F1 by `0.0309`, AP by `0.0329`, and AUC by `0.0132`. The 224 Grad-CAM grids still commonly activated near the joint line, but visible border/one-sided activations remained in both correct and failed cases; no anatomy-mask metric establishes a heatmap-localization improvement at 224. Therefore, changing to 224 does not solve the ROI/heatmap issue and must not replace the 384 production checkpoint.

### Production Grad-CAM Examples

The locked 384x384 evaluation exported predicted-class and true-class Grad-CAM galleries:

| KL grade | Correct examples | Failure examples |
| ---: | --- | --- |
| 0 | [success gallery](../archive/dense_net_121/runs/2026-07-30_09-03-29_paired_view_yolo_roi/assets/gradcam_success_grade_0.png) | [failure gallery](../archive/dense_net_121/runs/2026-07-30_09-03-29_paired_view_yolo_roi/assets/gradcam_failure_grade_0.png) |
| 1 | [success gallery](../archive/dense_net_121/runs/2026-07-30_09-03-29_paired_view_yolo_roi/assets/gradcam_success_grade_1.png) | [failure gallery](../archive/dense_net_121/runs/2026-07-30_09-03-29_paired_view_yolo_roi/assets/gradcam_failure_grade_1.png) |
| 2 | [success gallery](../archive/dense_net_121/runs/2026-07-30_09-03-29_paired_view_yolo_roi/assets/gradcam_success_grade_2.png) | [failure gallery](../archive/dense_net_121/runs/2026-07-30_09-03-29_paired_view_yolo_roi/assets/gradcam_failure_grade_2.png) |
| 3 | [success gallery](../archive/dense_net_121/runs/2026-07-30_09-03-29_paired_view_yolo_roi/assets/gradcam_success_grade_3.png) | [failure gallery](../archive/dense_net_121/runs/2026-07-30_09-03-29_paired_view_yolo_roi/assets/gradcam_failure_grade_3.png) |
| 4 | [success gallery](../archive/dense_net_121/runs/2026-07-30_09-03-29_paired_view_yolo_roi/assets/gradcam_success_grade_4.png) | [failure gallery](../archive/dense_net_121/runs/2026-07-30_09-03-29_paired_view_yolo_roi/assets/gradcam_failure_grade_4.png) |

## Base Production Training

Run: `2026-07-30 07:08:32 UTC`
Artifact: [`2026-07-30_07-08-32_original_224_ce_3stage`](../archive/dense_net_121/runs/2026-07-30_07-08-32_original_224_ce_3stage/)

Configuration: the production architecture, CE, sampler, augmentation, CLAHE `1.25 -> pad`, and three-stage schedule listed above, trained on published `224 x 224` crops. Published-crop test, `n=1,656`:

| Accuracy | QWK | Macro F1 | Macro AP | Macro AUC |
| ---: | ---: | ---: | ---: | ---: |
| 0.6697 | 0.8330 | 0.6800 | 0.7305 | 0.8980 |

Decision: use as initialization for paired-view production adaptation, not as the final production-domain result.

## Loss Comparison

Run: `2026-07-25 06:30:25 UTC`
Artifact: [`2026-07-25_06-30-25_final_noncanonical_loss_ablation`](../archive/dense_net_121/runs/2026-07-25_06-30-25_final_noncanonical_loss_ablation/)

Fixed configuration: natural orientation, full sampler, `384 x 384`, native linear-map research head, identical split/seed/augmentation, and 5+15+10 stages. Validation result:

| Loss arm | Accuracy | QWK | Macro F1 | Grade 1 recall | Macro AP | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CE | 0.6465 | **0.8083** | **0.6819** | 0.3856 | **0.7310** | selected |
| Ordinal PD-2 | 0.3862 | 0.6679 | 0.3352 | **0.8824** | 0.3521 | rejected; collapsed other classes |
| CE + 0.25 PD-2 | 0.6453 | 0.8066 | 0.6783 | 0.4314 | 0.7308 | rejected; no composite gain |

Selected CE locked-test result: Accuracy `0.6504`, QWK `0.8197`, macro F1 `0.6823`, AP `0.7309`, AUC `0.8935`.

## Preprocessing Comparison

Run: `2026-07-25 23:48:22 UTC`
Artifact: [`2026-07-25_23-48-22_preprocessing_quality_ablation`](../archive/dense_net_121/runs/2026-07-25_23-48-22_preprocessing_quality_ablation/)

Fixed configuration: CE, natural orientation, full sampler, same split, augmentation, schedule, and checkpoint rule. Validation result:

| Preprocessing arm | Epoch | Accuracy | QWK | Macro F1 | G1 recall | AP | AUC | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Raw -> pad | 29 | 0.6489 | 0.8117 | 0.6810 | 0.4444 | 0.7305 | 0.8899 | rejected |
| Pad -> CLAHE 2.0 | 30 | 0.6489 | 0.8110 | 0.6867 | 0.4314 | 0.7317 | 0.8881 | control |
| CLAHE 2.0 -> pad | 27 | 0.6598 | 0.8176 | 0.6879 | 0.4248 | 0.7341 | 0.8896 | rejected |
| CLAHE 1.25 -> pad | 30 | **0.6695** | **0.8274** | **0.7061** | **0.5294** | 0.7411 | 0.8951 | selected candidate |
| Percentile 1-99 -> pad | 27 | 0.6671 | 0.8242 | 0.6797 | 0.4444 | 0.7310 | **0.8964** | rejected |
| CLAHE 1.25 -> pad + acquisition robustness | 30 | 0.6477 | 0.8082 | 0.6930 | 0.4837 | **0.7458** | 0.8941 | rejected; lower QWK/F1 |

## Preprocessing Confirmation

Run: `2026-07-26 06:44:07 UTC`
Artifact: [`2026-07-26_06-44-07_preprocessing_confirmation`](../archive/dense_net_121/runs/2026-07-26_06-44-07_preprocessing_confirmation/)

| Arm | QWK | Macro F1 | Grade 1 recall | AP | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Pad -> CLAHE 2.0 | 0.8065 | **0.6878** | 0.4314 | 0.7316 | control |
| CLAHE 1.25 -> pad | **0.8142** | 0.6846 | **0.4837** | **0.7366** | retained for production training |

## Orientation Comparison

Run: `2026-07-25 00:34:38 UTC`
Artifact: [`2026-07-25_00-34-38_orientation_augmentation_ablation`](../archive/dense_net_121/runs/2026-07-25_00-34-38_orientation_augmentation_ablation/)

| Arm | QWK | Macro F1 | Grade 1 recall | AP | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Canonical baseline | **0.8196** | **0.6980** | **0.4575** | **0.7193** | validation winner, not adopted because production retains natural orientation |
| Natural orientation + flip | 0.8053 | 0.6820 | 0.3987 | 0.7119 | selected non-canonical arm; test QWK `0.8224` |
| Natural orientation + flip + mild affine | 0.8011 | 0.6800 | 0.3072 | 0.7067 | rejected |

## ROI Robustness Comparison

Run: `2026-07-28 04:58:51 UTC`
Artifact: [`2026-07-28_04-58-51_roi_robustness_ablation`](../archive/dense_net_121/runs/2026-07-28_04-58-51_roi_robustness_ablation/)

| Arm | QWK | Macro F1 | Grade 1 recall | AP | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Single-cutout control | **0.8177** | **0.6836** | **0.4641** | **0.7526** | retained |
| ROI geometry jitter | 0.7999 | 0.6797 | 0.4510 | 0.7371 | rejected |
| Geometry + acquisition augmentation | 0.8000 | 0.6625 | 0.4510 | 0.7329 | rejected |

No robustness candidate passed the classification and localization gates.

## YOLO Crop Expansion Comparison

Run: `2026-07-29 04:54:34 UTC`
Artifact: [`2026-07-29_04-54-34_yolo_crop_expansion_ablation`](../archive/dense_net_121/runs/2026-07-29_04-54-34_yolo_crop_expansion_ablation/)

Configuration: the same checkpoint and deterministic transform were evaluated across YOLO crop expansion factors. Result: `1.00` was the validation recommendation for that checkpoint. This was a diagnostic result, later superseded by paired-view training and the production `1.15` crop policy.

## Paired-View Comparison

Run: `2026-07-29 12:21:26 UTC`
Artifact: [`2026-07-29_12-21-26_paired_view_ablation`](../archive/dense_net_121/runs/2026-07-29_12-21-26_paired_view_ablation/)

Selected arm: `paired_expanded_x1_15`, epoch `5`. Published-domain QWK `0.8043`; target YOLO-domain QWK `0.7189`, macro F1 `0.6064`, AP `0.6551`. Decision: paired views substantially reduced the target-domain failure and established the method used by the later production run.

## Production-ROI Robustness Fine-Tune

Run: `2026-07-30 15:45:03 UTC`
Artifact: [`2026-07-30_15-45-03_production_roi_robustness`](../archive/dense_net_121/runs/2026-07-30_15-45-03_production_roi_robustness/)

| Arm | Accuracy | QWK | Macro F1 | AP | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Fixed-ROI production baseline | 0.5884 | 0.7405 | 0.6220 | 0.6665 | retained |
| Crop-jitter candidate, epoch 4 | **0.6186** | **0.7650** | **0.6451** | **0.6756** | rejected; aggregate Grad-CAM geometry regressed |

## Deep/GLCM Feature Fusion

Run: `2026-07-31 00:49:00 UTC`
Artifact: [`2026-07-31_00-49-00_glcm_fusion_comparison`](../archive/dense_net_121/runs/2026-07-31_00-49-00_glcm_fusion_comparison/)

Selected additive deep+GLCM arm, epoch `5`: Accuracy `0.6235`, QWK `0.7737`, macro F1 `0.6519`, Grade 1 recall `0.4379`, AP `0.6963`, AUC `0.8739`. Decision: not promoted because the predeclared multi-seed improvement gates were not met.

## Historical Single-Configuration Runs

These use older published-crop protocols and remain historical evidence, not production-domain comparisons.

| Timestamp UTC | Configuration | Accuracy | QWK | Macro AP | Status/artifact |
| --- | --- | ---: | ---: | ---: | --- |
| 2026-07-15 13:42:33 | CE baseline | 0.6691 | 0.8058 | 0.7009 | [completed](../archive/dense_net_121/runs/2026-07-15_13-42-33_ce_baseline/) |
| 2026-07-15 17:30:22 | CE + full sampler + minority augmentation + double cutout | 0.6594 | 0.8283 | 0.7287 | [completed](../archive/dense_net_121/runs/2026-07-15_17-30-22_ce_regularized/) |
| 2026-07-16 20:45:12 | focal CORN, low LR | 0.6087 | 0.7388 | 0.6775 | [underfit](../archive/dense_net_121/runs/2026-07-16_20-45-12_focal_corn_underfit/) |
| 2026-07-17 10:33:24 | focal CORN, corrected LR | 0.6612 | 0.8271 | 0.7280 | [completed](../archive/dense_net_121/runs/2026-07-17_10-33-24_focal_corn_optimized_lr/) |
| 2026-07-17 16:06:42 | focal CORN, corrected LR/patience | **0.6733** | **0.8394** | **0.7439** | [best historical published-crop metric](../archive/dense_net_121/runs/2026-07-17_16-06-42_focal_corn_optimized_lr_patience/) |
| 2026-07-17 22:15:13 | focal CORN, gradual unfreeze | 0.6498 | 0.7564 | 0.7059 | [invalid unfreeze implementation](../archive/dense_net_121/runs/2026-07-17_22-15-13_focal_corn_gradual_unfreeze/) |
| 2026-07-18 20:27:46 | focal CORN, moderated sampler | 0.6534 | 0.7624 | 0.7124 | [invalid unfreeze implementation](../archive/dense_net_121/runs/2026-07-18_20-27-46_focal_corn_moderated_sampler/) |
| 2026-07-18 22:03:35 | focal CORN, 384px | 0.6564 | 0.7796 | 0.7297 | [invalid unfreeze implementation](../archive/dense_net_121/runs/2026-07-18_22-03-35_focal_corn_384_resolution_frozen/) |
| 2026-07-20 12:36:36 | CORN, three-stage | 0.6715 | 0.8246 | 0.7337 | [completed](../archive/dense_net_121/runs/2026-07-20_12-36-36_corn/) |
| 2026-07-20 17:09:20 | final-layer Grad-CAM checkpoint | 0.6685 | 0.8223 | 0.7345 | [provenance incomplete](../archive/dense_net_121/runs/2026-07-20_17-09-20_final_layer_gradcam/) |
| 2026-07-21 15:07:17 | canonical CE + native-CAM head | 0.6534 | 0.8238 | 0.7311 | [completed](../archive/dense_net_121/runs/2026-07-21_15-07-17_canonical_final_linear_cam/) |

## Archive Notes

- [`2026-07-25_15-32-33_api_cam_localization_audit`](../archive/dense_net_121/runs/2026-07-25_15-32-33_api_cam_localization_audit/) and [`2026-07-27_gradcam_test_images_audit`](../archive/dense_net_121/runs/2026-07-27_gradcam_test_images_audit/) retain one representative montage each; unlabeled API images cannot establish accuracy.
- [`2026-07-29_roi_annotation_reference`](../archive/dense_net_121/runs/2026-07-29_roi_annotation_reference/) contains ROI examples, not a training run.
- A proposed unexecuted `512 x 512` notebook was removed because it had no result and was superseded by the production `384 x 384` configuration.
