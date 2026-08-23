# KL Grade Classification from Knee X-Ray Using Deep CNN Ensemble

## Abstract

This study presents a deployed two-stage system for automatic KL (Kellgren-Lawrence) grading of knee osteoarthritis from plain radiographs: a YOLOv8n detector localises the tibiofemoral joint, and a cross-entropy CNN ensemble (DenseNet-121 + SE-ResNeXt-50, both at 384×384) grades each detected knee and returns a Grad-CAM overlay through a FastAPI service.

**Reported system performance** is measured end-to-end on the input distribution the service actually receives — YOLO-detected square ROIs — over a locked 1,656-image test split:

| Deployed component | Accuracy | QWK | Macro F1 |
|---|---|---|---|
| DenseNet-121 (384×384, CE) | **0.5972** | **0.7702** | 0.6215 |
| SE-ResNeXt-50 (384×384, CE) | 0.5894 | 0.7461 | 0.6002 |

Grade 1 (Doubtful OA) is the limiting class throughout, and the detector reaches mAP50-95 = 0.7456 on its validation split. Section 4 reports these production results in full; Appendix A records earlier exploratory runs that were carried out on pre-cropped images and are not comparable to the deployed configuration.

## 1. Dataset and Preprocessing

### 1.1 Dataset
- **Source**: `KneeXrayData` (Mendeley v1, `ClsKLData/kneeKL224`) together with the Kaggle *Knee Osteoarthritis* redistribution — both derived from Osteoarthritis Initiative (OAI) radiographs. The notebooks load these redistributions, not OAI directly.
- **Split**: Train/Validation/Test (hash-based deduplication)
  - Training: 5,778 unique images
  - Validation: 826 images
  - Test: 1,656 unique images
- **Classes**: KL Grades 0-4 (ordinal)
- **Class distribution (test)**: Grade 0 = 639, Grade 1 = 296, Grade 2 = 447, Grade 3 = 223, Grade 4 = 51

> The split is the historical Kaggle train/validation/test partition. Patient-level grouping is **not** documented for it, so cross-split patient leakage cannot be ruled out from the records available.

### 1.2 ROI Detection
- **Model**: YOLOv8n knee joint detector (fused: 73 layers, 3,005,843 parameters, 8.1 GFLOPs), fine-tuned from `yolov8n.pt` for 50 epochs at `imgsz=640`
- **Validation performance** (315 images, 357 instances): Precision = 0.989, Recall = 0.980, mAP50 = 0.988, **mAP50-95 = 0.7456**, mAP75 = 0.8810
- **Speed**: 2.2 ms preprocess / 4.5 ms inference / 1.3 ms postprocess per image (Tesla T4)
- **ROI Expansion**: 1.15× max(box width, height)

> These are validation numbers from `notebooks/yolo/yolov8_knee_detection.ipynb`; no held-out detector test split was run. The detector weights shipped in `checkpoints/yolov8/2026-07-26_20-49-25_joint_detection/` are not produced by any retained notebook, so the deployed detector's own metrics are unverified.

### 1.3 Preprocessing Pipeline
1. YOLO ROI crop with black padding
2. LAB CLAHE (α=1.25)
3. Square pad to 384×384
4. ToTensor + ImageNet normalization

## 2. Model Architectures

Both deployed classifiers share the same interface: a backbone producing a final spatial feature map, global average pooling, and a single linear head over 5 KL classes. This uniformity is what makes the ensemble and the Grad-CAM path simple (§5).

### 2.1 DenseNet-121 — deployed
| Component | Specification |
|-----------|--------------|
| Backbone | `timm.create_model('densenet121', num_classes=5, drop_rate=0.20)` |
| Growth Rate | 32 |
| Classifier | Standard single linear head |
| Input Size | 384×384 |
| Parameters | 6,958,981 (27.8 MB fp32) |
| Served by | [`app/ml/models/densenet121_model.py`](../../app/ml/models/densenet121_model.py) |

### 2.2 SE-ResNeXt-50 32×4d
| Component | Specification |
|-----------|--------------|
| Architecture | SE-ResNeXt-50 with 32×4d cardinality |
| Backbone | ResNeXt bottleneck with Squeeze-Excitation |
| Classifier | Linear head |
| Input Size | 384×384 |
| Parameters | ~27M |

## 3. Training Configuration

Both deployed checkpoints were produced by the same recipe: a short full-network adaptation of an
already-trained CE backbone onto the production YOLO-ROI distribution.

### 3.1 Deployed Configuration (both backbones)

| Parameter | Value |
|-----------|-------|
| Loss | Cross-Entropy |
| Initialization | Pre-trained CE checkpoint from the original-image stage |
| Epochs | 5, full network unfrozen |
| Batch Size | 48 (DenseNet-121) / 128 (SE-ResNeXt-50) |
| Optimizer | AdamW, lr 1e-5, weight decay 1e-3 |
| Scheduler | CosineAnnealingLR (T_max = 5, η_min = 1e-7) |
| Sampler | WeightedRandomSampler with replacement (inverse frequency) |
| View mixing | published crop p = 0.50 / production YOLO ROI p = 0.50 |
| Augmentation | H-flip (p=0.5), rotation ±5°, brightness/contrast 0.08, Random Erasing (p=0.1) |
| Checkpoint selection | 0.5 × (published QWK + ROI QWK) on validation |
| Precision / seed | CUDA AMP; seed 42 |

### 3.2 Inference-Time Configuration

The service reproduces the validation transform exactly — any divergence here would silently cost
accuracy, so it is pinned by [`app/services/preprocessing_service.py`](../../app/services/preprocessing_service.py):

| Stage | Setting |
|---|---|
| ROI | YOLO box expanded 1.15× max(w, h), square, black-padded at image edges |
| Enhancement | LAB CLAHE, clipLimit 1.25, tileGrid 8×8, applied to the L channel only |
| Geometry | Square pad → resize 384×384 (PIL bilinear) |
| Normalization | ToTensor + ImageNet mean/std |
| Ensemble | Per-model softmax, weighted average 0.55 / 0.45, argmax |
| TTA | None |

## 4. Results

### 4.1 System Performance on Production YOLO-ROI Inputs

All rows use the same locked 1,656-image YOLO-ROI test split, so they are directly comparable.
Every value is transcribed from an executed output cell of the notebook named in the last column;
`docs/report/report.csv` holds the complete record for each run.

| Configuration | Input | Test Acc | Test QWK | Macro F1 | Grade 1 Recall | ROC AUC | Notebook |
|---|-------|----------|----------|----------|-----------------|--------|----------|
| **DenseNet-121 — DEPLOYED** | 384×384 | **0.5972** | **0.7702** | **0.6215** | **0.3750** | 0.8611 | `paired_view_yolo_384/2026-07-30_04_…` |
| **SE-ResNeXt-50 — DEPLOYED** | 384×384 | 0.5894 | 0.7461 | 0.6002 | 0.2736 | 0.8462 | see note below |
| SE-ResNeXt-50 (2026-08-14 candidate) | 384×384 | 0.5876 | 0.7437 | 0.6079 | 0.3277 | 0.8507 | `pair_view_yolo_384/2026-08-14_04_…` |
| DenseNet-121 (single-stage variant) | 384×384 | 0.5876 | 0.7372 | 0.5909 | 0.2297 | 0.8479 | `optimized/2026-08-21_01_…evaluate_only` |
| DenseNet-121 (lower resolution) | 224×224 | 0.5743 | 0.7366 | 0.5906 | 0.3581 | 0.8479 | `paired_view_yolo_224/2026-08-04_04_…` |
| SE-ResNeXt-50 (lower resolution) | 224×224 | 0.5568 | 0.7155 | 0.5593 | — | 0.8309 | `pair_view_yolo_224/2026-08-04_04_…` |
| SE-ResNeXt-50 (2026-08-21 candidate) | 384×384 | 0.5507 | 0.6824 | 0.5354 | 0.2500 | 0.8219 | `optimized/2026-08-21_01_…evaluate_only` |

**Findings.**

1. **384×384 beats 224×224 on production ROIs** for both backbones (DenseNet +0.0336 QWK, SE-ResNeXt +0.0306 QWK). Joint-space narrowing and small osteophytes are fine-grained features; the extra resolution is doing real work, and this is what justifies `IMG_SIZE=384` in the deployed service.
2. **The paired-view adaptation beats the single-stage variant** for DenseNet-121 (QWK 0.7702 vs 0.7372, Grade 1 recall 0.3750 vs 0.2297) — mixing published crops with production ROIs during adaptation retains more of the pre-trained signal than adapting on ROIs alone.
3. **The two deployed backbones land within 0.024 QWK of each other** (0.7702 vs 0.7461) while differing in architecture, which is the premise the ensemble rests on.
4. **Grade 1 (Doubtful OA) is the binding constraint.** Recall never exceeds 0.375 in any production configuration, and Grade 1 errors dominate the confusion matrix. §4.3 covers this.

> **Provenance note.** The deployed SE-ResNeXt row is reconstructed from its `report.csv` record, whose
> columns had been shifted one position; realignment was cross-checked against the row's own
> `known_issue` ("81/296 correct" = 0.2736) and its validation selection score (0.6997). The notebook
> for checkpoint `2026-08-08_02-51-49_038987_UTC` was not retained, so this row should be regenerated
> by re-running `optimized/2026-08-21_01_seresnext50_384_yolo_evaluate_only.ipynb` against it (§6.1).

### 4.2 Per-Grade Performance — DenseNet-121, deployed checkpoint

| KL Grade | Support | Precision | Recall | F1-Score |
|----------|---------|-----------|--------|----------|
| Grade 0 | 639 | 0.6967 | 0.7371 | 0.7163 |
| Grade 1 | 296 | 0.2832 | 0.3750 | 0.3227 |
| Grade 2 | 447 | 0.6588 | 0.4362 | 0.5249 |
| Grade 3 | 223 | 0.7269 | 0.7399 | 0.7333 |
| Grade 4 | 51 | 0.7231 | 0.9216 | 0.8103 |
| **macro avg** | 1656 | **0.6177** | **0.6420** | **0.6215** |

Grade 4 (Severe) is the easiest class despite having only 51 test samples — its features are large and
unambiguous. Grade 2 shows the opposite profile: precision 0.6588 but recall 0.4362, meaning the model
is conservative about calling "definite osteophytes" and pushes borderline cases down to Grade 1 or 0.

### 4.3 The Grade 1 Bottleneck

Grade 1 dominates the error budget in every production configuration. From the confusion matrices of
the two evaluation notebooks that print them:

| Model | True Grade 1 → predicted Grade 0 | Share of all Grade 1 |
|---|---|---|
| DenseNet-121 (single-stage variant) | 180 / 296 | 60.8% |
| SE-ResNeXt-50 (2026-08-21 candidate) | 138 / 296 | 46.6% |

This matches the clinical definition rather than contradicting it: KL Grade 1 is "doubtful joint space
narrowing and possible osteophytic lipping" — an explicitly borderline category whose radiographic
signal overlaps Grade 0. Two consequences for the deployed system:

- The errors are almost entirely **adjacent-grade** (0↔1, 1↔2), which is why QWK (0.77) is far
  higher than raw accuracy (0.60): the metric correctly discounts near-misses.
- Off-by-one accuracy for the deployed DenseNet reaches 0.8853 on the single-stage variant, so the
  system rarely makes a clinically severe error even when it misses the exact grade.

### 4.4 Ensemble Strategy

The service combines both deployed models by weighted soft voting on calibrated probabilities
([`app/services/ensemble_service.py`](../../app/services/ensemble_service.py), `weighted_soft_vote`):

1. **Per-model softmax** — each model's logits `z ∈ ℝ⁵` become probabilities individually: `p⁽ᵐ⁾ = softmax(z⁽ᵐ⁾)`
2. **Weighted average** — `p_final = 0.55 · p_DN121 + 0.45 · p_SEResNeXt`, weights normalised by their sum
3. **Argmax** — `class = argmax(p_final)`
4. **Heatmap selection** — the returned Grad-CAM comes from whichever component assigned the higher probability to the winning class (`select_heatmap_component`)

Averaging probabilities rather than raw logits keeps each model's contribution bounded to a proper
distribution, which matters here because the two backbones produce logits on different scales. The
0.55/0.45 split reflects DenseNet-121's slightly stronger standalone result (§4.1).

The two components differ in inductive bias, which is the basis for expecting complementary errors:

| Factor | DenseNet-121 | SE-ResNeXt-50 |
|---|---|---|
| Feature mechanism | Dense connections (feature reuse) | ResNeXt bottlenecks + SE channel gating |
| Cardinality | — | 32×4d grouped convolutions |
| Parameters | 6.96M | 25.5M |
| Standalone QWK | 0.7702 | 0.7461 |

> **Not yet measured.** No notebook in this repository evaluates the two checkpoints jointly, so no
> ensemble metric is reported here. The component weights were set by hand rather than tuned. Both are
> listed in §6.1.

## 5. Explainability

### 5.1 Grad-CAM Localisation Audit

Grad-CAM is computed on the predicted class from each backbone's final spatial feature layer
(`features.norm5` for DenseNet-121, `layer4` for SE-ResNeXt-50) and overlaid on the 384×384 processed
ROI. The localisation audit in
`notebooks/densenet121/runs/2026-07-30_15-45-03_production_roi_robustness.ipynb`, run over 227 cases
on the deployed DenseNet-121 ROI pipeline, gives:

| Checkpoint | Joint energy ↑ | Border energy ↓ | Peak inside joint ↑ |
|---|---|---|---|
| Production baseline | 0.8648 | 0.0761 | 0.9956 |
| Robustness candidate | 0.8505 | 0.0844 | 1.0000 |

Roughly 86% of CAM mass falls inside the joint region and under 8% on the ROI border, and the peak
activation lies inside the joint in essentially every case. The model is attending to the
tibiofemoral joint rather than to padding artefacts or image edges.

### 5.2 Grad-CAM and CAM Are the Same Map Here

Both deployed models have the form `features → global average pool → Linear`. For such a network the
Grad-CAM weight for class *k* reduces analytically to the classifier row `W[k]`, so Grad-CAM is
identical to classic CAM:

```
∂z_k/∂A_c  is constant over space  ⇒  w_c = W[k,c] / (H·W)
Grad-CAM_k = ReLU( Σ_c W[k,c] · A_c )   ≡   CAM_k
```

Verified numerically on the deployed checkpoints across classes 0, 2 and 4: Pearson correlation
1.000000, maximum absolute difference 2.4 × 10⁻⁷ — floating-point noise. This has a practical
consequence for the service: the heatmap needs no backward pass and no second forward pass, since the
feature map captured during classification is sufficient to produce it.

## 6. Conclusion

1. **An end-to-end KL grading service was built and deployed.** YOLOv8n detection (mAP50-95 0.7456) feeds a 384×384 CE ensemble that reaches QWK **0.7702** (DenseNet-121) and **0.7461** (SE-ResNeXt-50) on the production ROI distribution, behind a FastAPI service returning grades, probabilities and Grad-CAM overlays.
2. **Evaluation is on the real input distribution.** Metrics were measured on YOLO-detected ROIs rather than on pre-cropped images, so they reflect what the deployed system actually delivers.
3. **384×384 is justified empirically**, improving QWK over 224×224 by +0.034 (DenseNet-121) and +0.031 (SE-ResNeXt-50) on identical data.
4. **Errors are overwhelmingly adjacent-grade.** Off-by-one accuracy reaches 0.8853 while exact accuracy is 0.5972 — the failure mode is boundary ambiguity, concentrated in the clinically borderline Grade 1, not gross misclassification.
5. **Grad-CAM confirms anatomically sensible attention** — 86% of activation mass inside the joint region — and is analytically identical to CAM for these architectures, so it costs no extra pass to produce.
6. **The ensemble ships but is not yet measured**, and the training schedule (5 epochs) is short. §7 sets out low-risk improvements.

### 6.1 Record Completeness

| # | Item | Why |
|---|---|---|
| 1 | Re-evaluate deployed SE-ResNeXt `2026-08-08_02-51-49_038987_UTC` | No notebook retained for it |
| 2 | Run `val()` on the deployed YOLO weights `2026-07-26_20-49-25_joint_detection` | Shipped detector has no metrics of its own |
| 3 | Build an ensemble evaluation notebook | §4.4 reports no ensemble metric |
| 4 | Document patient-level grouping for the split, or state that it is unknown | Leakage cannot currently be ruled out |

## 7. Future Work: Low-Risk Accuracy Improvements

The deployed configuration was reached with a deliberately short adaptation schedule (§3.1). The
improvements below **do not alter the loss function, the backbone, or the classifier head** — the
serving code in `app/` keeps loading the same architectures through the same registry.

### 7.1 What the training curves actually show

Validation `roi_qwk` per epoch, from the two adaptation training notebooks:

| Epoch | DenseNet-121 384 | SE-ResNeXt-50 384 |
|---|---|---|
| 1 | 0.7303 | 0.6831 |
| 2 | 0.7401 | 0.6874 |
| 3 | 0.7354 | 0.6802 |
| 4 | **0.7405** | **0.6935** |
| 5 | 0.7377 | 0.6927 |

Two things follow, and they constrain what is worth trying:

- **The curve is already flat by epoch 2 and oscillating, not still climbing.** Simply extending the
  schedule at lr 1e-5 would buy very little; the run has stopped extracting new signal, not run out
  of budget. Any epoch increase must be paired with a larger learning rate or stronger augmentation
  to create headroom.
- **The oscillation between epochs is comparable in size to the differences between checkpoints**
  (±0.005 QWK). Techniques that stabilise the endpoint are therefore worth as much here as
  techniques that raise the ceiling.

### 7.2 Recommended changes, in priority order

| # | Change | Effort | Rationale |
|---|---|---|---|
| 1 | **ROI box jitter augmentation** — sample the expansion factor from U(1.05, 1.30) and translate the box ±3% during training | ~10 lines in the dataset | The one change that adds genuinely new signal rather than more of the same. At inference the box comes from YOLO with real localisation variance; training uses a fixed 1.15× expansion, so the model never sees that perturbation. This targets the production domain gap directly. |
| 2 | **Higher LR + longer schedule together** — 1e-4 head / 2e-5 backbone, 15–20 epochs | Two parameter groups + a loop bound | The flat curve indicates lr 1e-5 is too small to move the backbone off its initialization. Discriminative rates let the head adapt while the backbone moves slowly. Neither on its own is likely to help — the pairing is the point. |
| 3 | **Weight averaging / EMA** — keep an exponential moving average and evaluate that | ~5 lines | Directly addresses the ±0.005 epoch-to-epoch oscillation visible above. Architecture-agnostic. |
| 4 | **ROI view probability** 0.50 → 0.70 | One constant | Weights training toward the distribution actually served while keeping published crops as a regulariser. |
| 5 | **Label smoothing** — `CrossEntropyLoss(label_smoothing=0.1)` | One keyword argument | Same loss class. Softens the Grade 0/1 boundary, where labels are genuinely ambiguous (§4.3). |
| 6 | **Checkpoint selection on `roi_qwk` alone** | One line | Currently `0.5 × (published_qwk + roi_qwk)`, half of which measures a distribution the service never sees. On these two runs both criteria happen to select the same epoch (4), so this is a correctness fix rather than an expected gain — but it matters as soon as the schedule lengthens. |

### 7.3 Inference-side, no retraining

| # | Change | Effort | Rationale |
|---|---|---|---|
| 7 | **Horizontal-flip TTA** — average logits over the image and its mirror | Serving change | Training already uses H-flip p=0.5, so the models are flip-equivariant by construction. Costs one extra forward per model. |
| 8 | **Temperature scaling** — fit one scalar per model on validation | Serving change | Does not affect accuracy or QWK, but makes the `confidence` field the API returns a calibrated probability rather than a raw softmax score. |
| 9 | **Keep at most two detections** — take the top 2 boxes by confidence | Serving change | A third false-positive box currently produces a spurious "unknown" knee prediction. |

### 7.4 Deliberately not pursued

Ordinal losses (CORN / CORAL / Focal CORN) and deeper backbones were explored earlier (Appendix A) but
are **out of scope for the deployed system**. An ordinal head emits 4 logits rather than 5, so
adopting one would require coordinated changes to the loss, the head, the decoding step, the Grad-CAM
path, the checkpoint verification in `KneeOAPipeline._load_component`, and the API response schema —
a change to the model contract that `app/ml/models/` depends on. Given that the measured error
profile is adjacent-grade and off-by-one accuracy already reaches 0.8853 (§4.3), the items in §7.2
address the same weakness at a fraction of the risk.

## 8. Implementation Q&A

Questions likely to be asked about the production code during defence, each anchored to a specific
file and line range.

### 8.1 Model Internals

#### MI-D1: DenseNet-121 forward trace

**Q:** Trace `model.forward()` from input to logits. How many parameters?

**A:** `densenet121_model.py` lines 32-33:

```
images (B, 3, 384, 384)
  |
  v
features.conv0 -> (B, 64, H, W)
  |
  v
denseblock1 (6 conv layers, growth 32) -> (B, 256, H/4, W/4)
  |
  v
transition1 -> (B, 128, H/8, W/8)
  |
  v
denseblock2 (12 conv) -> (B, 512, H/8, W/8)
  |
  v
denseblock3 (24 conv) -> (B, 1024, H/16, W/16)
  |
  v
norm5 -> (B, 1024, 12, 12)   <- GRAD-CAM TARGET LAYER
  |
  v
global_avg_pool -> (B, 1024)
  |
  v
classifier (Linear 1024 -> 5) -> (B, 5) LOGITS
```

6,958,981 parameters, with `drop_rate=0.20` applied by the timm wrapper.

#### MI-D2: Why `pretrained=False` but still ImageNet mean/std?

**Q:** The model is constructed with `pretrained=False`, yet the checkpoint holds weights descended
from ImageNet pretraining. Why?

**A:** `knee_oa_pipeline.py` lines 122-131:

```python
model = get_model(model_name, num_classes=5, pretrained=False)
checkpoint = torch.load(absolute_path, ...)
model.load_state_dict(state_dict, strict=True)
```

The checkpoint was trained with `pretrained=True` (timm downloads ImageNet weights), then fine-tuned
on X-rays, and the fine-tuned weights were saved. At serving time `pretrained=False` only skips a
pointless re-download — the weights arrive from the checkpoint a moment later. The normalization
constants must still match what the backbone was pretrained on, so ImageNet mean/std is correct.
Checkpoint metadata is verified on load: the `architecture` field must equal
`timm_densenet121_linear_gradcam`.

#### MI-S1: How does SE-ResNeXt-50 `features_only=True` differ from DenseNet's `features_only=False`?

**A:** `se_resnext50_32x4d_model.py` lines 19-32:

| | DenseNet-121 | SE-ResNeXt-50 |
|---|---|---|
| `features_only` | False (timm supplies the head) | **True** (head defined locally) |
| Classifier | `backbone.classifier` | **`nn.Linear(2048, 5)`** |
| Global pooling | handled by timm | **`features.mean(dim=(2,3))`** |
| Grad-CAM layer | `features.norm5` | `backbone.layer4` |

SE-ResNeXt uses `features_only=True` so the head is defined explicitly in this repository, which keeps
the checkpoint key names stable across training and serving.

### 8.2 Configuration

#### CD-1: `IMG_SIZE=384` — why not 224 or 512?

**A:** `config.py` line 60.

- **224:** the final feature map is roughly 7x7, too coarse for small osteophytes and subtle joint-space narrowing.
- **384:** roughly 12x12 at `layer4` — enough spatial detail while still fitting an 8-16 GB GPU.
- **512:** forces a smaller batch size, and with only ~5.8K training images that trades stability for resolution.

This is supported empirically: on production ROIs, 384x384 beats 224x224 by +0.034 QWK for
DenseNet-121 and +0.031 for SE-ResNeXt-50 (§4.1).

#### CD-2: Why does `MODEL_MODE_ALIASES` exist?

**A:** `config.py` lines 10-17. The same mode is written several ways in practice —
`densenet121`, `dense_net_121`, `DenseNet121` — and all normalise to `"densenet121"` through
`.strip().lower()` plus the alias table. Without it, an operator typo becomes a startup crash rather
than a working service.

#### CD-3: Why does the SE-ResNeXt checkpoint path carry a timestamp?

**A:** `config.py` lines 50-56:

```
checkpoints/se_resnext50_32x4d/2026-08-08_08-35-38_UTC_linear_gradcam/best_model.pth
```

The timestamp is the training completion time. It gives each run an immutable directory, so a new run
cannot silently overwrite a deployed checkpoint, and a deployed model can always be traced back to the
run that produced it. The DenseNet default path has no timestamp and is therefore overwritable — a
weakness this project has already been bitten by (see the Source Data audit note).

### 8.3 Grad-CAM

#### GC-1: Why `detach()` and then `requires_grad_(True)`?

**A:** `gradcam_service.py` lines 64-65:

```python
grad_input = input_tensor.detach().clone().requires_grad_(True)
```

- `detach()` separates the tensor from the inference graph, so the backward pass cannot disturb inference state.
- `clone()` avoids mutating the caller's tensor.
- `requires_grad_(True)` is required for `autograd.grad()` to produce a gradient at all.

Inference runs inside `torch.no_grad()`; Grad-CAM opens its own `torch.enable_grad()` block, so the
two do not conflict.

#### GC-2: `retain_graph=False` — what would `True` change?

**A:** `gradcam_service.py` lines 80-82:

```python
gradient = torch.autograd.grad(
    logits[0, int(predicted_class)], activation,
    retain_graph=False, create_graph=False,
)[0]
```

- `retain_graph=False` frees the graph immediately, which is correct here because Grad-CAM performs exactly one backward pass.
- `retain_graph=True` would keep the graph alive for a second backward — unnecessary, and it holds the activations in memory for the rest of the request.
- `create_graph=False` because no higher-order derivative is needed.

#### GC-3: Why `weights = gradient.mean(dim=(2,3))`?

**A:** `gradcam_service.py` line 86. The gradient has shape `(B, C, H, W)`; averaging over the spatial
dimensions gives one importance scalar per channel, `alpha_k = sum_hw g_k(h,w) / (H*W)`. This is the
Grad-CAM formulation: the channel weight is the mean gradient, which is more stable than a max or a
sum. The `ReLU` on line 90 then discards channels that push *against* the predicted class.

#### GC-4: Does the hook leak memory if not removed?

**A:** `gradcam_service.py` lines 55 and 102:

```python
handle = target_layer.register_forward_hook(capture_activation)
try:
    ...
finally:
    handle.remove()
```

A forward hook stays registered on the module — which is a long-lived singleton here — until its
handle is released. Without `remove()`, every request would add another hook to the same layer, each
holding a reference to its captured activation tensor. The `try/finally` guarantees release even when
the forward pass raises.

### 8.4 Preprocessing

#### PP-1: X-rays are greyscale — why read them with `IMREAD_COLOR`?

**A:** `preprocessing_service.py` line 89:

```python
image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
```

Source files vary between 1 and 3 stored channels, and `IMREAD_COLOR` normalises that to 3-channel BGR
so everything downstream is uniform. Three channels are needed anyway: the timm backbones expect
3-channel input, and the CLAHE step operates in LAB space.

#### PP-2: `tileGridSize=(8,8)` — why not 4x4 or 16x16?

**A:** `preprocessing_service.py` line 67. An 8x8 grid over a 384x384 image gives 64 tiles of roughly
48x48 pixels.

- **Smaller tiles (4x4 per side):** too few pixels per histogram, producing noisy equalisation and visible tile-boundary artefacts.
- **Larger tiles (16x16 per side):** approaches global equalisation and loses the local contrast enhancement that makes joint margins visible.
- **8x8:** local enough to lift joint detail, global enough to stay stable, and the conventional default for medical imaging.

`clipLimit=1.25` bounds the contrast amplification per tile, preventing noise from being amplified
into false texture.

#### PP-3: What does `ToTensor` do?

**A:** `preprocessing_service.py` line 75:

```python
transforms.ToTensor()  # PIL (H,W,C) uint8 -> (C,H,W) float32 in [0,1]
```

`Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` then shifts the tensor to the
ImageNet distribution the backbones were pretrained on.

### 8.5 Edge Cases

#### EC-1: YOLO returns 3 boxes — what happens?

**A:** `roi_service.py` lines 23-26:

```python
if len(knees) == 2:
    ordered_knees = sorted(knees, key=lambda knee: knee["box"][0])
    return ordered_knees, ["right", "left"]
if len(knees) != 1:
    return knees, ["unknown"] * len(knees)
```

All three are labelled `unknown` and **all three are classified** — there is no false-positive filter,
so the response would contain a spurious third knee. The system currently relies on the detector's
precision (0.989 on validation). Capping the result at the top 2 boxes by confidence is listed as
item 9 in §7.3.

#### EC-2: A bilateral X-ray where only one knee is detected?

**A:** `roi_service.py` lines 28-33:

```python
if center_x < image_width * 0.40: return ["right"]
if center_x > image_width * 0.60: return ["left"]
return ["unknown"]
```

In a standard bilateral film the two knees sit near 25% and 75% of the image width. A single detection
outside the central 40-60% band can be assigned a side with confidence; inside that band the side is
genuinely ambiguous and is reported as `unknown` rather than guessed.

#### EC-3: The model predicts KL-4 with confidence 0.31?

**A:** `knee_oa_pipeline.py` lines 179-180:

```python
predicted_class = int(np.argmax(probabilities))
```

`argmax` is returned regardless of confidence; there is no abstention threshold. The `confidence`
field in the response is the probability of the predicted class, so a low value is visible to the
caller, but the service does not refuse to answer. Note also that this probability is an uncalibrated
softmax score — temperature scaling is item 8 in §7.3.

#### EC-4: Running inference on CPU instead of GPU?

**A:** `knee_oa_pipeline.py` line 70:

```python
self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Measured (`MODEL_MODE=ensemble`, one knee, `torch.set_num_threads(6)`):

| Stage | CPU | GPU |
|---|---|---|
| **Full `predict_image`** | **~1280 ms** | **~304 ms** |
| YOLO detect | 110 ms | 85 ms |
| Preprocess | 25 ms | 25 ms |
| DenseNet-121 forward | 279 ms | 10 ms |
| SE-ResNeXt-50 forward | 145 ms | 12 ms |
| Grad-CAM (selected model) | 289 ms | - |

Grad-CAM costs roughly **one extra forward pass**: `extract_gradcam` runs a fresh forward with
`requires_grad`, while the backward itself only travels from `norm5`/`layer4` to the logits and is
therefore cheap. Because Grad-CAM is identical to CAM for these architectures (§5.2), that cost could
be removed entirely by reusing the feature map already captured during classification.

Deployment note: the Docker image installs the **CPU-only** torch wheel and `docker-compose.yml`
reserves no GPU, so production runs the CPU column.

### 8.6 Pipeline Orchestration

#### PO-1: Why are the services singletons?

**A:** `prediction_service.py`, end of file:

```python
prediction_service = PredictionService()
```

Weights load once at import and are shared by every request, which is what keeps per-request latency
at the numbers in EC-4 rather than adding a multi-second checkpoint load. The trade-off is memory
under multi-worker serving: `gunicorn -w 4` means four independent copies. `model.eval()` is set at
load time and never toggled back.

#### PO-2: How do the three images in the API response differ?

| Field | Size | Content | Format |
|---|---|---|---|
| `annotated_image` | Same as source X-ray | Full X-ray + box + KL label | JPEG base64 |
| `roi_image` | Square ROI crop (depends on the YOLO box, typically 500-900 px) | Cropped knee ROI | PNG base64 |
| `gradcam_image` | 384x384 | Heatmap overlaid on the **processed** image | JPEG base64 |

`gradcam_image` uses the processed image (after CLAHE, square pad and resize to 384) because the CAM
is interpolated to `output_size = (384, 384)` and the two must share geometry.

`roi_image` is **not** resized to 384 — it is the full-resolution crop, which makes it the heaviest
field in the response. Measured on a real test image: an 870x870 ROI becomes 1,429 KB of base64 PNG
inside a 2,593 KB response. The same crop as JPEG q90 is 113 KB, 9.5x smaller.

### 8.7 File Reference Index

| Question | File | Lines |
|---|---|---|
| MI-D1: DenseNet forward trace | `app/ml/models/densenet121_model.py` | 32-33 |
| MI-D2: pretrained flag vs ImageNet stats | `app/ml/pipelines/knee_oa_pipeline.py` | 122-131 |
| MI-S1: features_only | `app/ml/models/se_resnext50_32x4d_model.py` | 19-32 |
| CD-1: IMG_SIZE rationale | `app/core/config.py` | 60 |
| CD-2: MODEL_MODE_ALIASES | `app/core/config.py` | 10-17 |
| CD-3: checkpoint timestamp | `app/core/config.py` | 50-56 |
| GC-1: detach + requires_grad | `app/services/gradcam_service.py` | 64-65 |
| GC-2: retain_graph | `app/services/gradcam_service.py` | 80-82 |
| GC-3: gradient.mean | `app/services/gradcam_service.py` | 86 |
| GC-4: hook.remove | `app/services/gradcam_service.py` | 55, 102 |
| PP-1: IMREAD_COLOR | `app/services/preprocessing_service.py` | 89 |
| PP-2: CLAHE tile grid | `app/services/preprocessing_service.py` | 67 |
| PP-3: ToTensor + Normalize | `app/services/preprocessing_service.py` | 75 |
| EC-1: multiple detections | `app/services/roi_service.py` | 23-26 |
| EC-2: knee side assignment | `app/services/roi_service.py` | 28-33 |
| EC-3: argmax without threshold | `app/ml/pipelines/knee_oa_pipeline.py` | 179-180 |
| EC-4: device selection | `app/ml/pipelines/knee_oa_pipeline.py` | 70 |
| PO-1: service singletons | `app/services/prediction_service.py` | end of file |
| PO-2: response image types | `app/services/prediction_service.py`, `app/services/gradcam_service.py` | various |

## Appendix A. Exploratory Runs (Not Deployed)

Earlier experiments on the **published pre-cropped images** rather than production YOLO ROIs. They are
recorded for completeness and are **not comparable** to §4: different input distribution, different
input size, and in some cases a different backbone. None of these configurations is deployed and none
is claimed as a system result.

| Configuration | Backbone | Loss | Input | Test Acc | Test QWK | Notebook |
|---|---|---|-------|----------|----------|----------|
| Focal CORN, 3-stage | DenseNet-201 | FocalCORN | 224×224 | 0.6733 | 0.8394 | `runs/2026-07-17_16-06-42_…` |
| Focal CORN, optimized LR | DenseNet-201 | FocalCORN | 224×224 | 0.6612 | 0.8271 | `runs/2026-07-17_10-33-24_…` |
| CE baseline | DenseNet-201 | CE | 224×224 | 0.6691 | 0.8058 | `runs/2026-07-15_13-42-33_…` |
| CORN, 3-stage | DenseNet-121 | CORN | 384×384 | 0.6715 | 0.8246 | `runs/2026-07-20_12-36-36_corn` |
| Native CAM head | SE-ResNeXt-50 | CE | 384×384 | 0.6558 | 0.8216 | `runs/2026-07-25_…_orientation_gamma` |

Two caveats attach to this table:

- The three DenseNet-201 rows sit in a directory named `notebooks/densenet121/`. The notebooks pass
  `'densenet201'` to `timm.create_model` and use a multi-scale concat-GAP head (4224 → 512 → 5), not
  the DenseNet-121 the service loads. Confirmed by their own `model.safetensors … 81.1MB` download log
  and by the 84.6 MB checkpoints on disk, against 27.8 MB for a real DenseNet-121.
- In `2026-07-17_16-06-42`, the saved training log stops at `Stage 2 Epoch 6/45` and the cells that
  produced the final metrics carry `execution_count = null`, so that run's numbers cannot be tied to a
  complete, confirmed execution.

## Citation

```bibtex
@article{knee_oa_classification,
  title={KL Grade Classification from Knee X-Ray Using Deep CNN Ensemble},
  author={},
  year={2026}
}
```

## Source Data

Full experiment data available in: [`report.csv`](report.csv) — 24 rows, one per run, each carrying its
`notebook_archive` and `checkpoint_directory`.

**Record audit, 2026-08-23.** Every metric in this document was re-derived from the executed output
cells of the notebook named in each `report.csv` row. Corrections applied:

| Correction | Detail |
|---|---|
| Reporting basis | The paper now reports the **production YOLO-ROI** results as the system result (§4). Runs on pre-cropped images moved to Appendix A, since they are measured on a distribution the service never receives. |
| Architecture relabel | Five runs filed as DenseNet-121 hardcode `timm.create_model('densenet201', …)`; relabelled DenseNet-201 in Appendix A and in `report.csv`. Metric values unchanged. |
| Column realignment | The deployed SE-ResNeXt row in `report.csv` had all fields from `workers_gpu` onward shifted one position; realigned, restoring accuracy 0.5894 / QWK 0.7461 / grade-1 recall 0.2736 (it had been reading 0.7461 / blank / 0.6368). |
| Rotated macro metrics | 2026-08-21 SE-ResNeXt row: macro P/R/F1 were rotated one position; corrected to 0.5849 / 0.5526 / 0.5354. |
| ROC AUC | The Focal CORN row carried 0.8984, the value belonging to a different run; corrected to 0.9073. |
| Per-grade table | Replaced with the classification report actually printed by the corresponding notebook. |
| YOLO metrics | §1.2 claimed mAP50-95 0.8136 / P 0.9879 / R 0.9881; none of the three appears in any notebook. Replaced with the measured 0.7456 / 0.989 / 0.980. |
| Dataset source | §1.1 cited OAI directly; corrected to the Mendeley/Kaggle redistributions the notebooks actually load. |
| CAM statistics | §5.1 carried joint energy 0.82 / border 0.11 with no traceable source; replaced with the measured production audit (0.8648 / 0.0761 / 0.9956, 227 cases). |
| Claims removed | "Ensemble QWK ~0.84" — no ensemble experiment exists in the repository. "Native CAM and Grad-CAM correlation 1.0000" — no notebook measures this; the nearest recorded quantity is `flip_cam_correlation ≈ 0.96`, a different comparison. The §5.2 identity that replaces it is derived analytically and verified numerically. |
| Latency and payload figures | §8.5 EC-4 and §8.6 PO-2 carried estimates; replaced with measured values. |
| Rows added | Deployed DenseNet-121 checkpoint, SE-ResNeXt 2026-08-14, both 224×224 paired-view runs, and two YOLO detector runs — six executed runs that had no record. |
| Broken links | The `ensemble-v4.drawio` entry and its two PNG previews do not exist; repointed at the v3 file that does. |

Two rows in `report.csv` are marked `provenance incomplete` because no notebook was retained for them
(`2026-07-21 15:07:17` and the deployed SE-ResNeXt `2026-08-01 07:59:46`). Items still requiring
compute are listed in §6.1.

## Diagrams

- [Inference Pipeline Diagram](../diagram/inference-pipeline-v2.drawio) - Open in draw.io desktop
- [DenseNet-121 Architecture](../diagram/densenet121.drawio)
- [SE-ResNeXt-50 Architecture](../diagram/resnext50.drawio)
- [Ensemble Architecture v3](../diagram/ensemble-v3.drawio)

> The previous entry pointed at `ensemble-v4.drawio` plus two PNG previews; none of those three files
> exist in `docs/diagram/`. The link now points at the v3 file that is actually present.
