# DenseNet-121 / SE-ResNeXt-50 — Focal CORN Research Pipelines

Standalone research variants of the production `pipeline/` notebooks. They
re-implement the same three-stage flow (base → paired-view ROI adaptation →
locked test evaluation) but switch the loss function from cross-entropy to
**Focal CORN** (Conditional Ordinal Regression for Normal-scores with focal
modulation).

| Stage | Production (`pipeline/`) | Research (`focal_corn/`) |
|-------|--------------------------|--------------------------|
| Loss | `F.cross_entropy` (5 logits, softmax) | `focal_corn_loss` (4 logits, sigmoid chain rule) |
| Head dim | `num_classes = 5` | `num_classes - 1 = 4` |
| Probabilities | `F.softmax` | `corn_probas` (chain rule) |
| Prediction | `argmax(softmax)` | `corn_label_from_logits` |
| Architecture tag | `timm_*_linear_gradcam` | `timm_*_linear_gradcam_ordinal` |
| Checkpoint `loss_type` | `ce` | `focal_corn` |

The shared loss/probability helpers live in
[`notebooks/_focal_corn_helpers.py`](../_focal_corn_helpers.py) and are pasted
into the relevant cells so the notebooks stay self-contained.

## Why this is research, not a deploy candidate

`paper.md` §7.4 explicitly documents why ordinal heads were kept out of the
deployed system:

> "An ordinal head emits 4 logits rather than 5, so adopting one would
> require coordinated changes to the loss, the head, the decoding step, the
> Grad-CAM path, the checkpoint verification in `KneeOAPipeline._load_component`,
> and the API response schema — a change to the model contract that
> `app/ml/models/` depends on."

The notebooks here let you measure whether the Focal CORN QWK gain
reported in Appendix A (DenseNet-201 + 224×224: 0.8394 vs 0.8058 for CE)
replicates on the **deployed** backbones (DenseNet-121 / SE-ResNeXt-50,
384×384) and on the **deployed** input distribution (YOLO ROI).

If a Focal CORN checkpoint is later promoted, `app/ml/models/` must be
extended first — the current 5-logit CE contract will not load it.

## Files

```
notebooks/densenet121/focal_corn/
├── 01_train_original_focal_corn.ipynb      # base training on published crops
├── 02_train_paired_roi_focal_corn.ipynb    # adaptation to YOLO ROI distribution
└── 03_evaluate_roi_test_focal_corn.ipynb   # locked test split evaluation

notebooks/seresnext50_32x4d/focal_corn/
├── 01_train_original_focal_corn.ipynb
├── 02_train_paired_roi_focal_corn.ipynb
└── 03_evaluate_roi_test_focal_corn.ipynb
```

Run order per backbone: `01` → `02` → `03`. Each stage writes
`SELECTED_CHECKPOINT.txt`; the next stage reads it. `02` does not load a
`01` checkpoint whose `loss_type` is `ce` — it expects `focal_corn`.