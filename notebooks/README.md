# Notebook Index

**Start here:** `<model>/pipeline/` holds the current, maintained workflow for each model — three
numbered notebooks that run in order and hand their selected checkpoint to the next stage.

| Pipeline | Notebooks |
| --- | --- |
| [`densenet121/pipeline/`](densenet121/pipeline/) | `01_train_original` → `02_train_paired_roi` → `03_evaluate_roi_test` |
| [`seresnext50_32x4d/pipeline/`](seresnext50_32x4d/pipeline/) | `01_train_original` → `02_train_paired_roi` → `03_evaluate_roi_test` |

Each pipeline folder has its own README with the required Drive layout, the configuration knobs, and
the promotion checklist.

The remaining folders are historical. Each model family uses the same layout: `runs/` contains one fixed configuration per notebook, `experiments/` contains multi-configuration comparisons, and `archive/` contains historical notebooks retained for provenance.

| Area | Contents |
| --- | --- |
| `densenet121/` | DenseNet-121 pipeline, runs, experiments, and archive. |
| `seresnext50_32x4d/` | SE-ResNeXt-50 pipeline, runs, experiments, and archive. |
| `datasets/` | Dataset download, analysis, and YOLO-ROI preparation. |
| `yolo/` | YOLO knee-detector training and command-line evaluation. |
| `comparison/` | Cross-model comparison notebooks. |
| `tools/` | Maintenance and model-inventory notebooks. |
| `paper/` | Paper-reproduction and reference notebooks. |

Completed result tables and figures belong in [`docs/report/`](../docs/report/). Reports link to the original notebook here; they do not store duplicate `.ipynb` files.
