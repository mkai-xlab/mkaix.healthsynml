# Joint-Guided Native-CAM Run Evaluation

## Run

The run directory was `2026-07-22_04-45-44_014517_UTC_joint_guided_cam`. It used three six-epoch fine-tuning arms and audited 227 validation cases. The test split was not loaded.

The run selected `joint_guided_005`, but it initialized from:

```text
last_model.pth
SHA-256: e492fd996cc1ab2d3d78954b8487978d695c2e53f3033947d3dd4a45357b1890
```

The production-best DenseNet checkpoint is different:

```text
best_model.pth
SHA-256: a8107b9cc7cc9242385f1facfcfc69c251f88697ed2df4dae8a43c1d66729b76
```

Therefore, this run is exploratory evidence and must not be promoted as the final model comparison.

## Metrics

| Arm | QWK | Macro F1 | Grade 1 recall | Joint peak rate | Lower-tibia energy |
| --- | ---: | ---: | ---: | ---: | ---: |
| CE control | 0.8070 | 0.6939 | 0.3922 | 0.9692 | 0.1688 |
| Joint guidance 0.02 | 0.8088 | 0.6954 | 0.3987 | 0.9736 | 0.1678 |
| Joint guidance 0.05 | 0.8072 | 0.6964 | 0.3987 | 0.9736 | 0.1662 |

The guidance arms slightly improved localization proxies and macro F1 relative to this run's control. They did not improve Grade 1 recall, and the differences are from one seed with a central-band proxy rather than expert anatomical annotations.

## CAM Decision

Native CAM remains the production method. The classifier creates one spatial map per KL grade and averages that map to obtain the grade logit. Consequently, the native map is part of the model's forward computation and is the most direct explanation for the predicted class.

Gradient Grad-CAM is added to the SE-ResNeXt notebook as a secondary diagnostic on the same final semantic feature map. It should be compared with occlusion/logit sensitivity and map agreement, not selected merely because a visual overlay appears smoother.

References:

- [Learning Deep Features for Discriminative Localization (CAM)](https://arxiv.org/abs/1512.04150)
- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
- [Grad-CAM++: Generalized Gradient-Based Visual Explanations](https://arxiv.org/abs/1710.11063)
- [Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292)
- [Tell Me Where to Look: Guided Attention Inference Network](https://arxiv.org/abs/1802.10171)

## Required Confirmation Run

The corrected DenseNet notebook now requires the production-best checkpoint and verifies its SHA-256 before training. Rerun the joint-guided arms from that checkpoint, then repeat them with at least two additional seeds before changing the application checkpoint. The SE-ResNeXt notebook now includes the same joint-guidance arm and a native-versus-Grad-CAM audit in its generated report.
