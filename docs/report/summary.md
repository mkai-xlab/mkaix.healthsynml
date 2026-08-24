# Run Summary

All 78 recorded configurations across 57 notebooks. Every row names exactly one notebook; ablation notebooks contribute one row per arm.
Full detail (40 columns) is in [`report.csv`](report.csv).

Regenerate with `python docs/report/build_summary.py` after editing `report.csv`.

## How to read this

**Row IDs** encode where a row came from: `DN-` DenseNet, `SE-` SE-ResNeXt, `YOLO-` detector,
`ORPHAN-` a checkpoint with no notebook. `-EXP-` is one arm of an ablation, `-RUN-` a standalone
run, `-TPL-` an unexecuted template.

**Split** is the single most important column. `test` means the locked hold-out split and is the
only number that can be quoted as a result. `validation` was used to choose epochs and settings,
so it is optimistic by construction and is not comparable to a test number. Never compare a
validation row against a test row.

**Two QWK values in one cell** (`0.79 (published) / 0.74 (ROI)`) mean the run was scored on both
the published crops and the YOLO ROI view. The ROI value is the one the deployed service sees.

**`—` means the notebook produced no such number.** It is never an estimate or a placeholder for
a value that exists elsewhere. Metrics are copied verbatim from executed output cells.

**Start with `Artifacts in production`.** Everything else is the evidence trail explaining why
those settings were chosen. `report.csv` carries the full configuration for every row;
[`audit_findings.md`](audit_findings.md) lists the discrepancies this audit turned up.

## Artifacts in production (3)

**A fourth production artifact is missing from this table on purpose:** the
deployed SE-ResNeXt-50 checkpoint (`checkpoints/se_resnext50_32x4d/2026-08-08_02-51-49_038987_UTC_paired_view_yolo_roi`) has no notebook anywhere in
the repository - no training run, no test evaluation. A row with no notebook would
break this table's own rule that every row names one executed notebook, so the gap
is documented in prose instead: see
[audit findings, section 4](audit_findings.md#4-the-deployed-se-resnext-50-has-no-notebook-at-all).

| ID | Model | Configuration | Input | Loss | Split | Acc | QWK | Macro F1 | G1 recall | Notebook |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DN-RUN-09 | DenseNet-121 | Paired-view adaptation: 50% published crop + 50% YOLO square ROI, roi_expansion 1.15 | 384x384 | Cross-Entropy (CE) | validation | — | 0.7969499 (published) / 0.7405191 (ROI) | 0.6706206 (published) / 0.6219707 (ROI) | — | [2026-07-30_03_train_densenet121_paired_view_yolo_384.ipynb](../../notebooks/densenet121/runs/paired_view_yolo_384/2026-07-30_03_train_densenet121_paired_view_yolo_384.ipynb) |
| DN-RUN-10 | DenseNet-121 | Locked YOLO-ROI test evaluation of the deployed checkpoint | 384x384 | n/a (inference only) | test | 0.5972222 | 0.7702197 | 0.6215203 | — | [2026-07-30_04_evaluate_densenet121_paired_view_yolo_gradcam_384.ipynb](../../notebooks/densenet121/runs/paired_view_yolo_384/2026-07-30_04_evaluate_densenet121_paired_view_yolo_gradcam_384.ipynb) |
| YOLO-02 | YOLOv8n | 100-epoch detector training, 640px | 640 | Ultralytics detection loss | validation | Precision 1.000 | mAP50-95 0.902 | — | — | [yolov8_knee_detection_cli.ipynb](../../notebooks/yolo/yolov8_knee_detection_cli.ipynb) |

## Locked test-split evaluations (5)

| ID | Model | Configuration | Input | Loss | Split | Acc | QWK | Macro F1 | G1 recall | Notebook |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SE-RUN-07 | SE-ResNeXt-50 32x4d | Locked YOLO-ROI test evaluation of the 384 paired-view checkpoint | 384x384 | n/a (inference only) | test | 0.5876 | 0.7437 | 0.6079 | — | [2026-08-14_04_evaluate_se_resnext50_yolo_roi_384_gradcam.ipynb](../../notebooks/seresnext50_32x4d/runs/pair_view_yolo_384/2026-08-14_04_evaluate_se_resnext50_yolo_roi_384_gradcam.ipynb) |
| DN-RUN-19 | DenseNet-121 | Locked YOLO-ROI test evaluation of the optimized 384 checkpoint | 384x384 | n/a (inference only) | test | 0.5876 | 0.7372 | 0.5909 | — | [2026-08-21_01_densenet121_384_yolo_evaluate_only.ipynb](../../notebooks/densenet121/runs/optimized/2026-08-21_01_densenet121_384_yolo_evaluate_only.ipynb) |
| DN-RUN-15 | DenseNet-121 | Locked YOLO-ROI test evaluation of the 224 paired-view checkpoint | 224x224 | n/a (inference only) | test | 0.5742754 | 0.7365876 | 0.5905817 | — | [2026-08-04_04_evaluate_densenet121_paired_view_yolo_gradcam_224.ipynb](../../notebooks/densenet121/runs/paired_view_yolo_224/2026-08-04_04_evaluate_densenet121_paired_view_yolo_gradcam_224.ipynb) |
| SE-RUN-04 | SE-ResNeXt-50 32x4d | Locked YOLO-ROI test evaluation of the 224 paired-view checkpoint | 224x224 | n/a (inference only) | test | 0.5567633 | 0.7155476 | 0.5593130 | — | [2026-08-04_04_evaluate_se_resnext50_paired_view_yolo_gradcam_224.ipynb](../../notebooks/seresnext50_32x4d/runs/pair_view_yolo_224/2026-08-04_04_evaluate_se_resnext50_paired_view_yolo_gradcam_224.ipynb) |
| SE-RUN-11 | SE-ResNeXt-50 32x4d | Locked YOLO-ROI test evaluation of the optimized 384 checkpoint | 384x384 | n/a (inference only) | test | 0.5507 | 0.6824 | 0.5354 | — | [2026-08-21_01_seresnext50_384_yolo_evaluate_only.ipynb](../../notebooks/seresnext50_32x4d/runs/optimized/2026-08-21_01_seresnext50_384_yolo_evaluate_only.ipynb) |

## Training runs (20)

| ID | Model | Configuration | Input | Loss | Split | Acc | QWK | Macro F1 | G1 recall | Notebook |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DN-RUN-05 | DenseNet-201 | 3-stage Focal CORN, optimized LR + reversed CORN task weights + longer patience | 224x224 | CORN -> CORN -> Focal CORN (gamma 2.0, alpha 0.25, task_weights [2.0,1.8,1.2,1.0]) | validation + test | 0.6733 | 0.8394 | 0.69 | — | [2026-07-17_16-06-42_focal_corn_optimized_lr_patience.ipynb](../../notebooks/densenet121/runs/2026-07-17_16-06-42_focal_corn_optimized_lr_patience.ipynb) |
| DN-RUN-08 | DenseNet-121 | Original published crops, CE, 3-stage (production original-stage recipe) | 384x384 | Cross-Entropy (CE) all stages | validation + test | 0.6697 | 0.8330 | 0.68 | — | [2026-07-30_07-08-32_original_224_ce_3stage.ipynb](../../notebooks/densenet121/runs/2026-07-30_07-08-32_original_224_ce_3stage.ipynb) |
| DN-RUN-02 | DenseNet-201 | Balanced sampler + minority augmentation + double cutout | 224x224 | Cross-Entropy (CE) | validation + test | 0.6594 | 0.8283 | 0.68 | — | [2026-07-15_17-30-22_ce_regularized.ipynb](../../notebooks/densenet121/runs/2026-07-15_17-30-22_ce_regularized.ipynb) |
| DN-RUN-04 | DenseNet-201 | 3-stage Focal CORN, optimized learning rates | 224x224 | CORN -> CORN -> Focal CORN (gamma 2.0, alpha 0.25, task_weights [1.0,1.2,2.0,3.5]) | validation + test | 0.6612 | 0.8271 | 0.67 | — | [2026-07-17_10-33-24_focal_corn_optimized_lr.ipynb](../../notebooks/densenet121/runs/2026-07-17_10-33-24_focal_corn_optimized_lr.ipynb) |
| DN-RUN-06 | DenseNet-121 | 3-stage CORN, 400x400 pad + 384 random crop | 384x384 (resize 400 -> crop 384) | CORN all stages (task_weights [2.0,1.8,1.2,1.0]) | validation + test | 0.6715 | 0.8246 | 0.68 | — | [2026-07-20_12-36-36_corn.ipynb](../../notebooks/densenet121/runs/2026-07-20_12-36-36_corn.ipynb) |
| SE-RUN-05 | SE-ResNeXt-50 32x4d | Original published crops upscaled to 384, CE, 3-stage | 384x384 | Cross-Entropy (CE) | validation + test | 0.6407005 | 0.8229052 | 0.6652209 | 0.3952703 | [2026-08-04_02_train_se_resnext50_original_384.ipynb](../../notebooks/seresnext50_32x4d/runs/pair_view_yolo_384/2026-08-04_02_train_se_resnext50_original_384.ipynb) |
| DN-RUN-07 | DenseNet-121 | 3-stage CORN, 400x400 pad + 384 crop, semantic Grad-CAM variant | 384x384 (resize 400 -> crop 384) | CORN all stages | validation + test | 0.6685 | 0.8223 | — | — | [2026-07-25_densenet201_noncanonical_loss_ablation_executed.ipynb](../../notebooks/densenet121/archive/2026-07-25_densenet201_noncanonical_loss_ablation_executed.ipynb) |
| SE-RUN-01 | SE-ResNeXt-50 32x4d | Natural orientation + training-only horizontal flip + mild gamma, native CAM, CE | 384x384 (resize 400 -> crop 384) | Cross-Entropy (CE) | validation + test | 0.6557971 | 0.8215502 | 0.6780896 | 0.4121622 | [2026-07-25_se_resnext50_native_cam_orientation_gamma.ipynb](../../notebooks/seresnext50_32x4d/runs/2026-07-25_se_resnext50_native_cam_orientation_gamma.ipynb) |
| DN-RUN-01 | DenseNet-201 | Baseline CE, no regularization, no balanced sampler | 224x224 | Cross-Entropy (CE) | validation + test | 0.6691 | 0.8058 | 0.67 | — | [2026-07-15_13-42-33_ce_baseline.ipynb](../../notebooks/densenet121/runs/2026-07-15_13-42-33_ce_baseline.ipynb) |
| SE-RUN-02 | SE-ResNeXt-50 32x4d | Original published crops at native 224, CE, 3-stage | 224x224 | Cross-Entropy (CE) | validation + test | 0.6262077 | 0.7968614 | 0.6464448 | 0.3445946 | [2026-08-04_02_train_se_resnext50_original_224.ipynb](../../notebooks/seresnext50_32x4d/runs/pair_view_yolo_224/2026-08-04_02_train_se_resnext50_original_224.ipynb) |
| DN-RUN-12 | DenseNet-121 | Original published crops at native 224, CE, 3-stage | 224x224 | Cross-Entropy (CE) all stages | validation + test | 0.6184 | 0.7931 | 0.62 | — | [2026-08-04_02_train_densenet121_original_224.ipynb](../../notebooks/densenet121/runs/paired_view_yolo_224/2026-08-04_02_train_densenet121_original_224.ipynb) |
| SE-RUN-06 | SE-ResNeXt-50 32x4d | Paired-view adaptation at 384: alternate view probability 0.50, YOLO ROI | 384x384 | Cross-Entropy (CE) | validation | — | 0.7912734 (published) / 0.6935232 (ROI) | 0.6654776 (published) / 0.5843468 (ROI) | — | [2026-08-14_03_train_se_resnext50_yolo_roi_384.ipynb](../../notebooks/seresnext50_32x4d/runs/pair_view_yolo_384/2026-08-14_03_train_se_resnext50_yolo_roi_384.ipynb) |
| DN-RUN-14 | DenseNet-121 | Paired-view adaptation at 224: 50% published + 50% YOLO ROI, roi_expansion 1.15 | 224x224 | Cross-Entropy (CE) | validation | — | 0.7778118 (published) / 0.7190776 (ROI) | 0.6514404 (published) / 0.5950677 (ROI) | — | [2026-08-04_03_train_densenet121_paired_view_yolo_224.ipynb](../../notebooks/densenet121/runs/paired_view_yolo_224/2026-08-04_03_train_densenet121_paired_view_yolo_224.ipynb) |
| DN-RUN-18 | DenseNet-121 | Paired published + YOLO ROI fine-tune at 384 from the optimized original checkpoint | 384x384 | plain CrossEntropyLoss | validation | — | 0.7758 (published) / 0.7116 (ROI) | — | — | [2026-08-16_02_densenet121_384_yolo_a100_ce.ipynb](../../notebooks/densenet121/runs/optimized/2026-08-16_02_densenet121_384_yolo_a100_ce.ipynb) |
| DN-RUN-16 | DenseNet-121 | A100-optimized original-crop training at native 224, plain CE | 224x224 | plain CrossEntropyLoss | validation | 0.6114 | 0.7725 | 0.6277 | — | [2026-08-16_01_densenet121_224_orig_a100_ce.ipynb](../../notebooks/densenet121/runs/optimized/2026-08-16_01_densenet121_224_orig_a100_ce.ipynb) |
| SE-RUN-03 | SE-ResNeXt-50 32x4d | Paired-view adaptation at 224: 50% published + 50% YOLO ROI, roi_expansion 1.15 | 224x224 | Cross-Entropy (CE) | validation | — | 0.7716616 (published) / 0.6868808 (ROI) | 0.6197724 (published) / 0.5528122 (ROI) | — | [2026-08-04_03_train_se_resnext50_paired_view_yolo_224.ipynb](../../notebooks/seresnext50_32x4d/runs/pair_view_yolo_224/2026-08-04_03_train_se_resnext50_paired_view_yolo_224.ipynb) |
| DN-RUN-11 | DenseNet-121 | YOLO-ROI-only fine-tune with random expansion 1.10-1.20 and <=5% centre shift | 384x384 | Cross-Entropy (CE) | validation | 0.6186441 | 0.7649809 | 0.6451166 | — | [2026-07-30_15-45-03_production_roi_robustness.ipynb](../../notebooks/densenet121/runs/2026-07-30_15-45-03_production_roi_robustness.ipynb) |
| SE-RUN-10 | SE-ResNeXt-50 32x4d | Paired published + YOLO ROI fine-tune at 384 from the optimized original checkpoint | 384x384 | plain CrossEntropyLoss | validation | — | 0.7521 (published) / 0.6888 (ROI) | — | — | [2026-08-16_02_seresnext50_384_yolo_a100_ce.ipynb](../../notebooks/seresnext50_32x4d/runs/optimized/2026-08-16_02_seresnext50_384_yolo_a100_ce.ipynb) |
| DN-RUN-03 | DenseNet-201 | 3-stage Focal CORN, under-fit baseline (low finetune LR 1e-5) | 224x224 | CORN -> CORN -> Focal CORN (gamma 2.0, alpha 0.25, task_weights [1.0,1.2,2.0,3.5]) | validation + test | 0.6087 | 0.7388 | 0.58 | — | [2026-07-16_20-45-12_focal_corn_underfit.ipynb](../../notebooks/densenet121/runs/2026-07-16_20-45-12_focal_corn_underfit.ipynb) |
| SE-RUN-08 | SE-ResNeXt-50 32x4d | A100-optimized original-crop training at native 224, plain CE | 224x224 | plain CrossEntropyLoss | validation | — | 0.7184 | — | — | [2026-08-16_01_seresnext50_224_orig_a100_ce.ipynb](../../notebooks/seresnext50_32x4d/runs/optimized/2026-08-16_01_seresnext50_224_orig_a100_ce.ipynb) |

## Ablation arms (31)

| ID | Model | Configuration | Input | Loss | Split | Acc | QWK | Macro F1 | G1 recall | Notebook |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DN-EXP-10 | DenseNet-121 | clahe1_25_then_pad (LAB CLAHE 1.25 before padding) | 384x384 | Cross-Entropy (CE) | validation | — | 0.8274009 | 0.7061268 | 0.5294118 | [2026-07-25_densenet121_preprocessing_clahe_ablation.ipynb](../../notebooks/densenet121/experiments/2026-07-25_densenet121_preprocessing_clahe_ablation.ipynb) |
| DN-EXP-11 | DenseNet-121 | percentile_1_99_then_pad (robust global 1-99 percentile windowing) | 384x384 | Cross-Entropy (CE) | validation | — | 0.8241785 | 0.6797090 | 0.4444444 | [2026-07-25_densenet121_preprocessing_clahe_ablation.ipynb](../../notebooks/densenet121/experiments/2026-07-25_densenet121_preprocessing_clahe_ablation.ipynb) |
| DN-EXP-02 | DenseNet-121 | natural_flip (natural laterality, RandomHorizontalFlip p=0.5, mild rotation) | 384x384 | Cross-Entropy (CE) | validation + test | 0.6539855 | 0.8224187 | 0.6801236 | 0.4189189 | [2026-07-25_densenet121_laterality_augmentation_ablation.ipynb](../../notebooks/densenet121/experiments/2026-07-25_densenet121_laterality_augmentation_ablation.ipynb) |
| DN-EXP-04 | DenseNet-121 | ce (standard five-class cross-entropy) | 384x384 | Cross-Entropy (CE) | validation + test | 0.6503623 | 0.8196540 | 0.6822715 | 0.3952703 | [2026-07-25_densenet121_natural_orientation_loss_ablation.ipynb](../../notebooks/densenet121/experiments/2026-07-25_densenet121_natural_orientation_loss_ablation.ipynb) |
| DN-EXP-01 | DenseNet-121 | canonical_baseline (mirror right knees, no random hflip, mild rotation) | 384x384 | Cross-Entropy (CE) | validation | — | 0.8196 | 0.6980 | 0.4575 | [2026-07-25_densenet121_laterality_augmentation_ablation.ipynb](../../notebooks/densenet121/experiments/2026-07-25_densenet121_laterality_augmentation_ablation.ipynb) |
| DN-EXP-15 | DenseNet-121 | single_cutout_control (no geometry jitter, no acquisition variation) | 384x384 | Cross-Entropy (CE) | validation | — | 0.8177 | 0.6836 | 0.4641 | [2026-07-28_densenet121_roi_robustness_ablation.ipynb](../../notebooks/densenet121/experiments/2026-07-28_densenet121_roi_robustness_ablation.ipynb) |
| DN-EXP-09 | DenseNet-121 | clahe2_then_pad (LAB CLAHE 2.0 before padding) | 384x384 | Cross-Entropy (CE) | validation | — | 0.8175507 | 0.6878950 | 0.4248366 | [2026-07-25_densenet121_preprocessing_clahe_ablation.ipynb](../../notebooks/densenet121/experiments/2026-07-25_densenet121_preprocessing_clahe_ablation.ipynb) |
| DN-EXP-14 | DenseNet-121 | clahe1_25_then_pad (confirmation, trained from scratch) | 384x384 | Cross-Entropy (CE) | validation | — | 0.8142283 | 0.6846063 | 0.4836601 | [2026-07-26_densenet121_clahe_order_confirmation.ipynb](../../notebooks/densenet121/experiments/2026-07-26_densenet121_clahe_order_confirmation.ipynb) |
| DN-EXP-07 | DenseNet-121 | raw_then_pad (no contrast enhancement) | 384x384 | Cross-Entropy (CE) | validation | — | 0.8117465 | 0.6810129 | 0.4444444 | [2026-07-25_densenet121_preprocessing_clahe_ablation.ipynb](../../notebooks/densenet121/experiments/2026-07-25_densenet121_preprocessing_clahe_ablation.ipynb) |
| DN-EXP-08 | DenseNet-121 | current_pad_then_clahe2 (deployed order: square pad then LAB CLAHE 2.0) | 384x384 | Cross-Entropy (CE) | validation | — | 0.8109745 | 0.6866649 | 0.4313725 | [2026-07-25_densenet121_preprocessing_clahe_ablation.ipynb](../../notebooks/densenet121/experiments/2026-07-25_densenet121_preprocessing_clahe_ablation.ipynb) |
| DN-EXP-12 | DenseNet-121 | clahe1_25_then_pad_acquisition_robust (CLAHE 1.25 + random gamma/blur/noise) | 384x384 | Cross-Entropy (CE) | validation | — | 0.8082430 | 0.6930339 | 0.4836601 | [2026-07-25_densenet121_preprocessing_clahe_ablation.ipynb](../../notebooks/densenet121/experiments/2026-07-25_densenet121_preprocessing_clahe_ablation.ipynb) |
| DN-EXP-21 | DenseNet-121 | base_unadapted (untouched base checkpoint, no fine-tuning) | 384x384 | Cross-Entropy (CE) | validation | 0.644068 | 0.806742 | 0.682205 | 0.483660 | [2026-07-29_densenet121_paired_view_yolo_ablation.ipynb](../../notebooks/densenet121/experiments/2026-07-29_densenet121_paired_view_yolo_ablation.ipynb) |
| DN-EXP-06 | DenseNet-121 | ce_plus_ordinal_pd2 (CE + 0.25 * normalized PD-2) | 384x384 | CE + 0.25*Ordinal PD-2 | validation | — | 0.8066 | 0.6783 | 0.4314 | [2026-07-25_densenet121_natural_orientation_loss_ablation.ipynb](../../notebooks/densenet121/experiments/2026-07-25_densenet121_natural_orientation_loss_ablation.ipynb) |
| DN-EXP-13 | DenseNet-121 | current_pad_then_clahe2 (confirmation, trained from scratch) | 384x384 | Cross-Entropy (CE) | validation | — | 0.8064971 | 0.6877679 | 0.4313725 | [2026-07-26_densenet121_clahe_order_confirmation.ipynb](../../notebooks/densenet121/experiments/2026-07-26_densenet121_clahe_order_confirmation.ipynb) |
| DN-EXP-22 | DenseNet-121 | published_control (5 epochs on published crops only) | 384x384 | Cross-Entropy (CE) | validation | 0.647700 | 0.805012 | 0.684013 | 0.457516 | [2026-07-29_densenet121_paired_view_yolo_ablation.ipynb](../../notebooks/densenet121/experiments/2026-07-29_densenet121_paired_view_yolo_ablation.ipynb) |
| DN-EXP-24 | DenseNet-121 | paired_expanded_x1_15 (50% published crop + 50% 1.15x expanded YOLO crop) | 384x384 | Cross-Entropy (CE) | validation | 0.646489 | 0.804346 | 0.679096 | 0.483660 | [2026-07-29_densenet121_paired_view_yolo_ablation.ipynb](../../notebooks/densenet121/experiments/2026-07-29_densenet121_paired_view_yolo_ablation.ipynb) |
| SE-EXP-02 | SE-ResNeXt-50 32x4d | final_native_cam_ordinal_soft_label (Gaussian ordinal soft-label CE, sigma 0.70) | 384x384 (resize 400 -> crop 384) | Ordinal soft-label CE (Gaussian sigma 0.70) | validation | 0.617433 | 0.803113 | 0.657431 | 0.424837 | [se_resnext50_32x4d_ce_vs_ordinal_loss_native_cam_ablation.ipynb](../../notebooks/seresnext50_32x4d/experiments/se_resnext50_32x4d_ce_vs_ordinal_loss_native_cam_ablation.ipynb) |
| DN-EXP-23 | DenseNet-121 | paired_raw_x1_00 (50% published crop + 50% raw YOLO crop) | 384x384 | Cross-Entropy (CE) | validation | 0.639225 | 0.802230 | 0.671061 | 0.483660 | [2026-07-29_densenet121_paired_view_yolo_ablation.ipynb](../../notebooks/densenet121/experiments/2026-07-29_densenet121_paired_view_yolo_ablation.ipynb) |
| DN-EXP-03 | DenseNet-121 | natural_flip_mild_affine (natural laterality, hflip, affine 7deg/3%/0.95-1.05/3deg shear) | 384x384 | Cross-Entropy (CE) | validation | — | 0.8011 | 0.6800 | 0.3072 | [2026-07-25_densenet121_laterality_augmentation_ablation.ipynb](../../notebooks/densenet121/experiments/2026-07-25_densenet121_laterality_augmentation_ablation.ipynb) |
| DN-EXP-17 | DenseNet-121 | roi_geometry_plus_acquisition (geometry jitter + gamma/blur/noise) | 384x384 | Cross-Entropy (CE) | validation | — | 0.8000 | 0.6625 | 0.4510 | [2026-07-28_densenet121_roi_robustness_ablation.ipynb](../../notebooks/densenet121/experiments/2026-07-28_densenet121_roi_robustness_ablation.ipynb) |
| DN-EXP-16 | DenseNet-121 | roi_geometry_jitter (scale-down + translate on square canvas, never crops anatomy) | 384x384 | Cross-Entropy (CE) | validation | — | 0.7999 | 0.6797 | 0.4510 | [2026-07-28_densenet121_roi_robustness_ablation.ipynb](../../notebooks/densenet121/experiments/2026-07-28_densenet121_roi_robustness_ablation.ipynb) |
| SE-EXP-03 | SE-ResNeXt-50 32x4d | full_inverse (sampler power 1.0) | 384x384 (resize 400 -> crop 384) | Cross-Entropy (CE) | validation | — | 0.7948 | 0.6446 | 0.3725 | [se_resnext50_32x4d_native_cam_sampler_strength_ablation.ipynb](../../notebooks/seresnext50_32x4d/experiments/se_resnext50_32x4d_native_cam_sampler_strength_ablation.ipynb) |
| SE-EXP-01 | SE-ResNeXt-50 32x4d | final_native_cam_ce (standard hard-label cross-entropy) | 384x384 (resize 400 -> crop 384) | Cross-Entropy (CE) | validation | 0.621065 | 0.793681 | 0.658106 | 0.366013 | [se_resnext50_32x4d_ce_vs_ordinal_loss_native_cam_ablation.ipynb](../../notebooks/seresnext50_32x4d/experiments/se_resnext50_32x4d_ce_vs_ordinal_loss_native_cam_ablation.ipynb) |
| SE-EXP-04 | SE-ResNeXt-50 32x4d | sqrt_inverse (sampler power 0.5) | 384x384 (resize 400 -> crop 384) | Cross-Entropy (CE) | validation | — | 0.7895 | 0.6496 | 0.3072 | [se_resnext50_32x4d_native_cam_sampler_strength_ablation.ipynb](../../notebooks/seresnext50_32x4d/experiments/se_resnext50_32x4d_native_cam_sampler_strength_ablation.ipynb) |
| SE-EXP-05 | SE-ResNeXt-50 32x4d | no_sampler (ordinary shuffled training) | 384x384 (resize 400 -> crop 384) | Cross-Entropy (CE) | validation | — | 0.7737 | 0.6348 | 0.1961 | [se_resnext50_32x4d_native_cam_sampler_strength_ablation.ipynb](../../notebooks/seresnext50_32x4d/experiments/se_resnext50_32x4d_native_cam_sampler_strength_ablation.ipynb) |
| DN-EXP-26 | DenseNet-121 | deep_glcm_additive (CNN + additive 6-feature GLCM logits, learned alpha) | 384x384 | unweighted Cross-Entropy (CE) | validation | 0.628329 (SD 0.013962) | 0.773399 (SD 0.009455) | 0.650411 | — | [2026-07-31_densenet121_glcm_feature_fusion_comparison.ipynb](../../notebooks/densenet121/experiments/2026-07-31_densenet121_glcm_feature_fusion_comparison.ipynb) |
| DN-EXP-25 | DenseNet-121 | deep_control (CNN only, no GLCM branch) | 384x384 | unweighted Cross-Entropy (CE) | validation | 0.633979 (SD 0.009404) | 0.771947 (SD 0.007236) | 0.656453 | — | [2026-07-31_densenet121_glcm_feature_fusion_comparison.ipynb](../../notebooks/densenet121/experiments/2026-07-31_densenet121_glcm_feature_fusion_comparison.ipynb) |
| DN-EXP-05 | DenseNet-121 | ordinal_pd2 (Chen et al. five-logit adjustable ordinal loss, normalized) | 384x384 | Ordinal PD-2 | validation | — | 0.6679 | 0.3352 | 0.8824 | [2026-07-25_densenet121_natural_orientation_loss_ablation.ipynb](../../notebooks/densenet121/experiments/2026-07-25_densenet121_natural_orientation_loss_ablation.ipynb) |
| DN-EXP-18 | DenseNet-121 | YOLO crop expansion x1.00 (raw production box) | 384x384 | n/a (inference only) | validation | 0.238499 | 0.220636 | 0.240907 | 0.000000 | [2026-07-29_densenet121_yolo_crop_expansion_ablation.ipynb](../../notebooks/densenet121/experiments/2026-07-29_densenet121_yolo_crop_expansion_ablation.ipynb) |
| DN-EXP-19 | DenseNet-121 | YOLO crop expansion x1.15 | 384x384 | n/a (inference only) | validation | 0.243341 | 0.212473 | 0.235535 | 0.006536 | [2026-07-29_densenet121_yolo_crop_expansion_ablation.ipynb](../../notebooks/densenet121/experiments/2026-07-29_densenet121_yolo_crop_expansion_ablation.ipynb) |
| DN-EXP-20 | DenseNet-121 | YOLO crop expansion x1.30 | 384x384 | n/a (inference only) | validation | 0.208232 | 0.148199 | 0.198118 | 0.000000 | [2026-07-29_densenet121_yolo_crop_expansion_ablation.ipynb](../../notebooks/densenet121/experiments/2026-07-29_densenet121_yolo_crop_expansion_ablation.ipynb) |

## Detector training (1)

| ID | Model | Configuration | Input | Loss | Split | Acc | QWK | Macro F1 | G1 recall | Notebook |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO-01 | YOLOv8n | 50-epoch detector training, 640px | 640 | Ultralytics detection loss | validation | Precision 0.989 | mAP50-95 0.745 | — | — | [yolov8_knee_detection.ipynb](../../notebooks/yolo/yolov8_knee_detection.ipynb) |

## Incomplete, stale, or never executed (15)

| ID | Model | Configuration | Input | Loss | Split | Acc | QWK | Macro F1 | G1 recall | Notebook |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DN-RUN-13 | DenseNet-121 | Declared img_size 384; outputs are the 224 run's | 384x384 (declared in source) | Cross-Entropy (CE) | none (stored outputs do not belong to this configuration) | — | — | — | — | [2026-08-04_02_train_densenet121_original_384.ipynb](../../notebooks/densenet121/runs/paired_view_yolo_384/2026-08-04_02_train_densenet121_original_384.ipynb) |
| DN-RUN-17 | DenseNet-121 | Declared: original crops upscaled to 384, plain CE, 3-stage | 384x384 (224 crops upscaled) | plain CrossEntropyLoss | none | — | — | — | — | [2026-08-16_01_train_densenet121_original_224_ordinal_optimized.ipynb](../../notebooks/densenet121/runs/optimized/2026-08-16_01_train_densenet121_original_224_ordinal_optimized.ipynb) |
| DN-TPL-01 | DenseNet-121 | Pipeline template: stage 1 original-crop training | — | — | — | — | — | — | — | [01_train_original.ipynb](../../notebooks/densenet121/pipeline/01_train_original.ipynb) |
| DN-TPL-02 | DenseNet-121 | Pipeline template: stage 2 paired-ROI adaptation | — | — | — | — | — | — | — | [02_train_paired_roi.ipynb](../../notebooks/densenet121/pipeline/02_train_paired_roi.ipynb) |
| DN-TPL-03 | DenseNet-121 | Pipeline template: stage 3 locked ROI test evaluation | — | — | — | — | — | — | — | [03_evaluate_roi_test.ipynb](../../notebooks/densenet121/pipeline/03_evaluate_roi_test.ipynb) |
| DN-TPL-04 | DenseNet-121 | Focal CORN experiment template: stage 1 | — | — | — | — | — | — | — | [01_train_original_focal_corn.ipynb](../../notebooks/densenet121/focal_corn/01_train_original_focal_corn.ipynb) |
| DN-TPL-05 | DenseNet-121 | Focal CORN experiment template: stage 2 | — | — | — | — | — | — | — | [02_train_paired_roi_focal_corn.ipynb](../../notebooks/densenet121/focal_corn/02_train_paired_roi_focal_corn.ipynb) |
| DN-TPL-06 | DenseNet-121 | Focal CORN experiment template: stage 3 evaluation | — | — | — | — | — | — | — | [03_evaluate_roi_test_focal_corn.ipynb](../../notebooks/densenet121/focal_corn/03_evaluate_roi_test_focal_corn.ipynb) |
| SE-RUN-09 | SE-ResNeXt-50 32x4d | Declared: original crops upscaled to 384, 3-stage | 384x384 | plain CrossEntropyLoss | none (run never completed) | — | — | — | — | [2026-08-16_01_train_se_resnext50_original_224_ordinal_optimized.ipynb](../../notebooks/seresnext50_32x4d/runs/optimized/2026-08-16_01_train_se_resnext50_original_224_ordinal_optimized.ipynb) |
| SE-TPL-01 | SE-ResNeXt-50 32x4d | Pipeline template: stage 1 original-crop training | — | — | — | — | — | — | — | [01_train_original.ipynb](../../notebooks/seresnext50_32x4d/pipeline/01_train_original.ipynb) |
| SE-TPL-02 | SE-ResNeXt-50 32x4d | Pipeline template: stage 2 paired-ROI adaptation | — | — | — | — | — | — | — | [02_train_paired_roi.ipynb](../../notebooks/seresnext50_32x4d/pipeline/02_train_paired_roi.ipynb) |
| SE-TPL-03 | SE-ResNeXt-50 32x4d | Pipeline template: stage 3 locked ROI test evaluation | — | — | — | — | — | — | — | [03_evaluate_roi_test.ipynb](../../notebooks/seresnext50_32x4d/pipeline/03_evaluate_roi_test.ipynb) |
| SE-TPL-04 | SE-ResNeXt-50 32x4d | Focal CORN experiment template: stage 1 | — | — | — | — | — | — | — | [01_train_original_focal_corn.ipynb](../../notebooks/seresnext50_32x4d/focal_corn/01_train_original_focal_corn.ipynb) |
| SE-TPL-05 | SE-ResNeXt-50 32x4d | Focal CORN experiment template: stage 2 | — | — | — | — | — | — | — | [02_train_paired_roi_focal_corn.ipynb](../../notebooks/seresnext50_32x4d/focal_corn/02_train_paired_roi_focal_corn.ipynb) |
| SE-TPL-06 | SE-ResNeXt-50 32x4d | Focal CORN experiment template: stage 3 evaluation | — | — | — | — | — | — | — | [03_evaluate_roi_test_focal_corn.ipynb](../../notebooks/seresnext50_32x4d/focal_corn/03_evaluate_roi_test_focal_corn.ipynb) |

## Utility notebooks (3)

| ID | Model | Configuration | Input | Loss | Split | Acc | QWK | Macro F1 | G1 recall | Notebook |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UTIL-01 | n/a (data preparation) | Build original 224 crops, full bilateral PNGs, and the YOLO square-ROI train/val/test splits | — | — | — | — | — | — | — | [01_prepare_original_and_yolo_roi_datasets.ipynb](../../notebooks/datasets/01_prepare_original_and_yolo_roi_datasets.ipynb) |
| UTIL-02 | n/a (analysis) | Count images per split and per class for the KL and YOLO-ROI datasets | — | — | — | — | — | — | — | [dataset_analysis.ipynb](../../notebooks/tools/dataset_analysis.ipynb) |
| UTIL-03 | n/a (housekeeping) | Inspect and organize Google Drive model files | — | — | — | — | — | — | — | [organize_google_drive_models.ipynb](../../notebooks/tools/organize_google_drive_models.ipynb) |

