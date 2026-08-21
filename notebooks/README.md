# Notebook Index

Use the notebook folders by purpose. Each model family uses the same layout: `runs/` contains one fixed configuration per notebook, `experiments/` contains multi-configuration comparisons, `production/` contains the production-training workflow, and `archive/` contains historical notebooks retained for provenance.

| Area | Contents |
| --- | --- |
| `densenet121/` | DenseNet-121 runs, experiments, production workflow, and archive. |
| `seresnext50_32x4d/` | SE-ResNeXt-50 runs, experiments, and archive. |
| `datasets/` | Dataset download, analysis, and YOLO-ROI preparation. |
| `yolo/` | YOLO knee-detector training and command-line evaluation. |
| `comparison/` | Cross-model comparison notebooks. |
| `tools/` | Maintenance and model-inventory notebooks. |
| `paper/` | Paper-reproduction and reference notebooks. |

Completed result tables and figures belong in [`docs/report/`](../docs/report/). Reports link to the original notebook here; they do not store duplicate `.ipynb` files.
