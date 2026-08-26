# Notebook-to-Record Audit

Every `.ipynb` under [`../../notebooks/`](../../notebooks/) was opened and read individually:
its configuration cells, its executed output cells, and — where a checkpoint survives on disk —
its stored metadata via `torch.load`. Ablation notebooks were split so that **each compared
configuration is its own row** in [`report.csv`](report.csv), pointing back at the single file
it came from.

**Coverage: 57 notebooks → 78 rows, each row backed by exactly one executed notebook. No notebook is unmapped, and no row lacks a notebook.**

Metrics are copied verbatim from executed cells. Where a notebook produced no number, the cell
reads `—`. Nothing in these files is estimated, inferred, or reconstructed.

---

## 1. Five "DenseNet-121" runs are actually DenseNet-201

Five notebooks under `notebooks/densenet121/runs/` call:

```python
self.backbone = timm.create_model('densenet201', pretrained=pretrained,
                                  features_only=True, out_indices=(2, 3, 4))
total_channels = 512 + 1792 + 1920     # DenseNet-201 stage widths
```

`TrainingConfig.model_name = "densenet121"` is a dead variable — it is never passed to
`create_model`. Three independent confirmations agree:

| Evidence | Value | DenseNet-121 | DenseNet-201 |
| --- | --- | --- | --- |
| `create_model` literal | `'densenet201'` | — | yes |
| `model.safetensors` download in the log | 81.1 MB | 32.3 MB | yes |
| stage channels | 512+1792+1920 | 512+1024+1024 | yes |

Affected rows, now labelled **DenseNet-201**: `DN-RUN-01` … `DN-RUN-05`
(2026-07-15 13-42-33, 2026-07-15 17-30-22, 2026-07-16 20-45-12, 2026-07-17 10-33-24,
2026-07-17 16-06-42).

**This includes the paper's headline result.** `DN-RUN-05` (accuracy 0.6733, QWK 0.8394) is a
DenseNet-201, not a DenseNet-121. `paper.md` must attribute it accordingly.

`DN-RUN-06` (2026-07-20 12-36-36, accuracy 0.6715 / QWK 0.8246) **is** a genuine DenseNet-121 —
it calls `create_model('densenet121')`, uses channels 512+1024+1024, and downloads 32.3 MB.

## 2. A correction to an earlier audit pass

`notebooks/densenet121/archive/2026-07-25_densenet201_noncanonical_loss_ablation_executed.ipynb`
is **DenseNet-121**, despite its filename. Verified: the class is `DenseNet121Model`, the call is
`create_model('densenet121')`, `total_channels = 512 + 1024 + 1024`, and the download is 32.3 MB.
Only the markdown title and one `print` say "DenseNet-201" — stale text copied from the
DenseNet-201 template. The notebook also contains **no ablation arms** despite its name; it is a
single run.

An earlier pass on this record relabelled that row DenseNet-201 from the filename alone. That was
wrong and is reversed here (`DN-RUN-07`). The lesson: filenames and titles in this repository are
not reliable; only the executed code is.

## 3. One notebook's outputs do not belong to its configuration

`notebooks/densenet121/runs/paired_view_yolo_384/2026-08-04_02_train_densenet121_original_384.ipynb`
has stored outputs that are **byte-for-byte identical** to the 224 notebook's — both serialize to
6,452,442 characters — while the sources differ only in `img_size = 224` → `384`.

The configuration was edited to 384 and the notebook was never re-executed. It displays the 224
run's numbers (accuracy 0.6184 / QWK 0.7931) under a 384 filename. Recorded as `DN-RUN-13`
with `record_type: stale_outputs` and **no metrics**. It must be re-run or have its outputs cleared.

## 4. The deployed SE-ResNeXt-50 has no notebook at all

`checkpoints/se_resnext50_32x4d/2026-08-08_02-51-49_038987_UTC_paired_view_yolo_roi/best_model.pth`
is the SE-ResNeXt-50 the application serves. Its metadata reads:

```
architecture   seresnext50_32x4d_linear_gradcam
head_type      linear_after_global_average_pool
cam_method     post_hoc_gradcam
loss_type      ce
epoch          4
paired_view_probability 0.5
roi_expansion  1.15
robust_selection 0.6990949665349357
```

Every notebook under `notebooks/seresnext50_32x4d/` was searched: **none references 2026-08-08.**
There is no training record and no test evaluation for the checkpoint in production. The sibling
`2026-08-01_05-29-59_660657_UTC` (native-CAM head, robust_selection 0.6997054) is in the same
position. Both are recorded as `ORPHAN-01` / `ORPHAN-02` with `record_type: orphan_checkpoint`.

The closest *measured* SE-ResNeXt result is `SE-RUN-07` — a **different** checkpoint
(2026-08-14 23-59-46): test accuracy 0.5876, QWK 0.7437, macro F1 0.6079.

**To close this gap:** run
`notebooks/seresnext50_32x4d/runs/optimized/2026-08-21_01_seresnext50_384_yolo_evaluate_only.ipynb`
with `RUN_DIR` pointed at the 2026-08-08 checkpoint. It is a parameterised evaluation-only
notebook, so this is a path change and one execution.

Three older DenseNet-121 archive checkpoints are in the same position — a `.pth` file survives with
a `validation_metrics` dict inside it, but no notebook produced that dict and none evaluates it on
test:

| Checkpoint | Validation QWK (from checkpoint metadata) | Notes |
| --- | ---: | --- |
| `densenet121/2026-07-27_04-05-44_natural_orientation_ce_gradcam` | 0.8071397 | ancestor of the 2026-07-29 base checkpoint used by the ROI ablations |
| `densenet121/archive/2026-07-21_15-07-17_..._canonical_final_linear_cam` | 0.8000991 | this is the checkpoint the old (pre-audit) report.csv attributed a test QWK of 0.8238 to — that number appears in no notebook and no checkpoint |
| `densenet121/archive/2026-07-23_01-31-37_..._canonical_final_linear_cam_production` | 0.8138684 | archived; superseded by the natural-orientation line |

None of these five orphan checkpoints (the two SE-ResNeXt ones above plus these three) appear as
rows in `report.csv` — a row with no notebook would contradict the file's own rule that every row
names one executed notebook. They are recorded here instead, so the gap in provenance is documented
without being disguised as a notebook-backed result.

## 5. A macro-recall value was reported as macro F1

`SE-RUN-11` (`2026-08-21_01_seresnext50_384_yolo_evaluate_only.ipynb`) prints:

```
Macro Recall    0.5526
Macro F1        0.5354
```

Any record quoting **macro F1 = 0.5526** for this run has copied the macro-recall column.
The correct macro F1 is **0.5354**.

## 6. The YOLO numbers in paper.md match neither detector run

`paper.md` §1.2 claims mAP50-95 = 0.8136, Precision = 0.9879, Recall = 0.9881. Actual outputs:

| Notebook | Val images | Instances | P | R | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `yolov8_knee_detection.ipynb` (50 ep) | 315 | 357 | 0.989 | 0.980 | 0.988 | **0.745** |
| `yolov8_knee_detection_cli.ipynb` (100 ep) | 58 | 117 | 1.000 | 0.991 | 0.995 | **0.902** |

Neither run produces 0.8136 / 0.9879 / 0.9881. Quote one run verbatim and name its split.
Note that the CLI run's better-looking numbers come from a **58-image** validation split, so they
are far less stable than the 315-image run's.

## 7. Runs that produced no usable result

| Row | Notebook | What happened |
| --- | --- | --- |
| `DN-RUN-17` | `optimized/2026-08-16_01_train_densenet121_original_224_ordinal_optimized.ipynb` | `KeyboardInterrupt` during Stage 1, epoch 1/5. No metrics. |
| `SE-RUN-09` | `optimized/2026-08-16_01_train_se_resnext50_original_224_ordinal_optimized.ipynb` | Stopped at Stage 2, epoch 10/15. Final "best selection" cell never ran; no checkpoint recorded. |
| `DN-RUN-13` | `paired_view_yolo_384/2026-08-04_02_train_densenet121_original_384.ipynb` | Outputs stale (see §3). |
| `DN-TPL-01…06`, `SE-TPL-01…06` | `pipeline/` and `focal_corn/` | Never executed: 0 execution counts, 0 outputs. Templates only. |

All twelve `pipeline/` and `focal_corn/` notebooks are clean templates. **The Focal CORN
experiment has not been run** — no Focal CORN result exists anywhere in the repository.

## 8. Truncated training logs

Three notebooks saved a final test result but a truncated epoch history:

| Row | Log stops at | Budget |
| --- | --- | --- |
| `DN-RUN-02` | Standard epoch 19 | 30 |
| `DN-RUN-03` | Stage 2 epoch 16 | 45 |
| `DN-RUN-05` | Stage 2 epoch 6 | 45 |

`DN-RUN-05` is the paper's headline run, so its epoch-by-epoch history is not reproducible from
the stored notebook, although its final test cell did execute. The metrics are real; the training
curve is not recoverable.

## 9. Epoch budgets were misrecorded

The three-stage Focal CORN runs use `stage1/2/3 = 5/25/15`, a **45-epoch** budget — not 30.
Affected: `DN-RUN-03`, `DN-RUN-04`, `DN-RUN-05`, `DN-RUN-06`, `DN-RUN-07`.

## 10. Two low-resolution ablations must not be read as accuracy

`DN-EXP-18…20` (YOLO crop expansion) report accuracy around 0.21–0.24 and QWK 0.15–0.22. These
are **not** model accuracy. The notebook evaluates a 384-trained checkpoint against
`DetKneeData/H5` images that are only **256×320**, far off-distribution, and states explicitly
that the experiment selects crop *policy* only. The rows carry this caveat in `notes`.

---

## What is solid

- The **deployed DenseNet-121** is fully traceable end to end:
  `DN-RUN-08` (original 384 CE 3-stage) → `DN-RUN-09` (paired-view adaptation, epoch 4,
  robust_selection 0.7217104) → `DN-RUN-10` (locked test: accuracy 0.5972222, QWK 0.7702197,
  macro F1 0.6215203, macro AUC 0.8611089, n = 1656). The checkpoint's stored
  `robust_selection` matches the notebook's epoch-4 value exactly.
- **Preprocessing is justified by evidence.** CLAHE 1.25 before padding won a six-arm ablation
  (`DN-EXP-10`, selection 0.7780713) and was re-confirmed from scratch in a clean two-arm re-run
  (`DN-EXP-14`, selection 0.7636995) after an interrupted-optimizer confound was found in the
  first study. Both checkpoints on disk match their notebooks' validation metrics exactly.
- **The 50/50 paired-view + 1.15× recipe is justified.** `DN-EXP-21` shows the unadapted model
  collapsing on production YOLO crops (QWK 0.806742 → 0.220636); `DN-EXP-24` recovers it
  (yolo_x1_15 QWK 0.718937) while staying within 0.01 of baseline on the published domain.
- **The class sampler is justified.** Removing it drops SE-ResNeXt Grade-1 recall to 0.1961
  (`SE-EXP-05`) versus 0.3725 with full inverse-frequency (`SE-EXP-03`).
- **Two negative results are properly recorded rather than buried:** GLCM texture fusion was
  rejected (`DN-EXP-26`, learned α collapsed to 0.018, `promote_to_locked_holdout: false`), and
  the ordinal soft-label gain over CE was **not** statistically significant (`SE-EXP-02`,
  QWK difference +0.009324, 95% CI −0.005974 to +0.024627).

## Open items, in priority order

1. **Evaluate the deployed SE-ResNeXt-50** (§4) — the production model currently has no measured
   test performance. One notebook execution with a changed path.
2. **Correct `paper.md`**: the DenseNet-201 attribution (§1), the YOLO detector numbers (§6),
   the macro-F1/macro-recall swap (§5), and the 45-epoch budgets (§9).
3. **Re-run or clear** `2026-08-04_02_train_densenet121_original_384.ipynb` (§3).
4. **No ensemble evaluation exists.** No notebook computes the DenseNet + SE-ResNeXt ensemble, so
   any ensemble figure in the report is unmeasured and should be removed until it is run.
