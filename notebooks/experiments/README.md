# Experiment Notebooks

Use this directory for controlled research only. The production-ready DenseNet
training notebook remains at `notebooks/production/dense_net_121_production.ipynb`.

| Directory | Contents |
| --- | --- |
| `densenet121/augmentation` | Cutout and orientation augmentation studies. |
| `densenet121/loss` | CE, CORN, and ordinal-loss comparisons. |
| `densenet121/preprocessing` | CLAHE, padding, and image-quality studies. |
| `densenet121/roi` | YOLO crop size, paired-view, and published-versus-YOLO workflows. Version 2 supersedes version 1. |
| `densenet121/heatmaps` | Grad-CAM/native-CAM localization experiments. |
| `seresnext50_32x4d` | Sampling, loss, and heatmap studies for SE-ResNeXt-50. |
| `efficientnet` | EfficientNet scale and CAM candidates. |
| `data_pipeline` | KneeXrayData download, audit, and YOLO ROI construction. |
| `model_comparison` | Cross-model comparisons such as Grad-CAM versus native CAM. |

Completed evidence belongs in `docs/report/<model>/`; do not overwrite a
completed notebook. Give new experimental notebooks a descriptive name ending
in `_ablation.ipynb` or `_comparison.ipynb`, and archive its timestamped
run directory and metrics with the report.
