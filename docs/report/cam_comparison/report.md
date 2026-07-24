# Grad-CAM and Native-CAM Controlled Comparison
This report records the exact two-model validation comparison completed on 2026-07-24 and distinguishes visual equivalence from operational preference.

## Run: 2026-07-24 01:12:36.714882 UTC

| Field | Value |
| --- | --- |
| Models | DenseNet-121 and SE-ResNeXt50-32x4d |
| Data | Validation only; test not read |
| Audit | 227 images: 50 per grade except all 27 Grade 4 images |
| Methods | Final-layer Grad-CAM and native class map for the same target class |
| Bootstrap | 2,000 paired resamples |

### Results

| Model | Method | Joint Energy | Border Energy | Anatomy Score | Occlusion Correlation | Mean Map Difference | Maximum Map Difference |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DenseNet-121 | Native CAM | 0.79972 | 0.13080 | 0.56750 | 0.71089 | 0.00014 | 0.001359 |
| DenseNet-121 | Grad-CAM | 0.79923 | 0.13101 | 0.56652 | 0.71095 | 0.00014 | 0.001359 |
| SE-ResNeXt50 | Native CAM | 0.86903 | 0.07690 | 0.70425 | 0.65282 | 0.00008 | 0.000592 |
| SE-ResNeXt50 | Grad-CAM | 0.86920 | 0.07682 | 0.70451 | 0.65280 | 0.00008 | 0.000592 |

Map correlation was `1.0000` for both models. The analytical gradient-weight error and the maximum difference between Grad-CAM and bias-free CAM were both zero. DenseNet native CAM had a tiny anatomy-score advantage while Grad-CAM had a tiny occlusion advantage; SE-ResNeXt showed the opposite pattern. The predeclared decision was `no_demonstrated_superiority` for both models.

### Checkpoint Resolution Caveat
The notebook accepted any filename containing `best_model`, then sorted matching paths. This selected `stage2_best_model.pth` for both architectures rather than the intended final `best_model.pth`. The exact model-level localization values must therefore be rerun after changing the resolver to require `path.name == "best_model.pth"`.

This issue does not undermine the main method conclusion. For a 1x1 linear class-map head followed by global average pooling, final-layer Grad-CAM analytically reduces to bias-free CAM. The experiment's zero implementation errors and near-identical images confirm that relationship for both architectures.

### Decision
Use native CAM in production because it provides the same practical heatmap with one forward pass, no backward pass, less memory, and simpler inference. Do not claim that native CAM is anatomically superior. Switching to Grad-CAM will not repair lateral or border-focused evidence because both methods expose the same learned feature map.

CAM and Grad-CAM were introduced as weakly supervised/post-hoc localization methods ([Zhou et al., 2016](https://doi.org/10.1109/CVPR.2016.319); [Selvaraju et al., 2017](https://doi.org/10.1109/ICCV.2017.74)). Saliency sanity checks show why visual plausibility alone is insufficient ([Adebayo et al., 2018](https://arxiv.org/abs/1810.03292)). Knee-OA work similarly warns that attention maps do not guarantee causal radiographic evidence ([Tiulpin et al., 2020](https://doi.org/10.3390/diagnostics10110932)).

### Archived Notebook

- [Executed two-model comparison](2026-07-24_01-12-36_gradcam_vs_native_cam.ipynb)
