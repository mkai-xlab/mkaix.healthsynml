# HealthSync ML Diagram Index

The report-facing AI figures are stored in [`ai_model/`](ai_model/). The three polished pages share one editable Draw.io source; every PNG embeds the diagram XML and can be reopened directly in Draw.io.

| Figure | Report focus | Deliverables |
| --- | --- | --- |
| 1. KL grading system | YOLO joint localization, production preprocessing, parallel classifiers, weighted soft voting, final KL grade, and the separate Grad-CAM explanation path | [PNG](ai_model/01_kl_grading_system.drawio.png) |
| 2. Ensemble architecture | DenseNet-121 and SE-ResNeXt-50 32x4d internals, five-class probability vectors, weights `0.55` and `0.45`, and final `argmax` | [PNG](ai_model/02_ensemble_architecture.drawio.png) |
| 3. Grad-CAM mechanism | Model target layers, predicted-class gradient, channel weighting, ReLU, upsampling, and ROI overlay | [PNG](ai_model/03_gradcam_mechanism.drawio.png) |
| 4. DenseNet-121 detail | Dense blocks, transitions, tensor sizes, dense connectivity, `features.norm5`, global pooling, dropout, linear head, CE logits, and inference softmax | [PNG](ai_model/04_densenet121_architecture.drawio.png) |
| 5. SE-ResNeXt-50 detail | Grouped residual stages, squeeze-excitation gate, `layer4`, five-map head, spatial-mean logits, CE, softmax, and Grad-CAM boundary | [PNG](ai_model/05_se_resnext50_architecture.drawio.png) |
| 6. Production preprocessing | YOLO threshold, 1.15x square geometry, padding, CLAHE, resize, normalization, safeguards, and training-only augmentation branch | [PNG](ai_model/06_production_preprocessing.drawio.png) |
| 7. Training and validation | Paired views, inverse-frequency sampler, three-stage base training, full-network adaptation, validation selector, and locked evaluation | [PNG](ai_model/07_training_validation_lifecycle.drawio.png) |

Sources: [three-page overview Draw.io](ai_model/healthsync_kl_ai_architecture.drawio), [four-page detail Draw.io](ai_model/healthsync_kl_ai_details.drawio), and [tldraw whiteboard](ai_model/healthsync_kl_ai_whiteboard.tldr).

## Important Technical Labels

- DenseNet-121 Grad-CAM target: `backbone.features.norm5`, tensor shape `B x 1024 x 12 x 12` for a `384 x 384` input.
- SE-ResNeXt-50 32x4d Grad-CAM target: `backbone.layer4`, tensor shape `B x 2048 x 12 x 12` for a `384 x 384` input.
- Ensemble grading fuses class probabilities only; feature maps and Grad-CAM images are not averaged.
- The SE-ResNeXt checkpoint architecture identifier remains `final_native_cam_ce`, but the API explanation is post-hoc predicted-class Grad-CAM.
- The report-facing figures intentionally omit anatomy-gate validation and focus on KL grading plus Grad-CAM generation.

The broader [`healthsync_ml/`](healthsync_ml/) set remains available for detailed API, training, validation, CI, and deployment documentation. The older `knee_oa_ensemble_system.*` files are retained as historical artifacts.
