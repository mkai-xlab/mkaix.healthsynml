# Knee Osteoarthritis Configurable Native-CAM API

FastAPI inference service for Kellgren-Lawrence grades 0-4. The inference path combines YOLOv8 knee-joint detection with an environment-selected DenseNet-121, SE-ResNeXt-50, EfficientNet-B0, or the production DenseNet/SE-ResNeXt ensemble.

The complete training, evaluation, native-CAM, ensemble, and deployment record
is in [the three-model KL system document](docs/three_model_kl_system.md).

All classifiers are inference-only native-CAM models. Production ensemble mode combines DenseNet and SE-ResNeXt five-class softmax vectors using normalized `0.55/0.45` weights. EfficientNet-B0 remains available as a standalone comparison mode but is excluded from the production vote because its completed standalone run did not beat the other two models and no labeled paired ensemble validation has shown that it adds value.

For each ensemble prediction, the service measures the predicted-grade map from both active models on that exact ROI. A map passes the anatomy gate when joint energy is at least `0.55`, border and lower-tibia energy are at most `0.25`, and its peak lies inside the broad joint band. Among passing maps, the service maximizes predicted-grade class support multiplied by the per-case anatomy score. If neither map passes, it renders the best available map and emits a warning. This prevents class agreement alone from forcing an obviously misplaced map.

## Runtime Contract

- DenseNet checkpoint: `checkpoints/densenet121/best_model.pth`
- SE-ResNeXt checkpoint: `checkpoints/se_resnext50_32x4d/best_model (1).pth`
- EfficientNet-B0 checkpoint: `checkpoints/efficientnet_b0/best_model.pth`
- YOLO checkpoint: `checkpoints/yolov8/best.pt`
- Preprocessing: laterality canonicalization, square padding, CLAHE, resize to 400, center crop to 384, ImageNet normalization
- Right knee convention: right ROIs are horizontally mirrored before classification
- Classification: selected single-model softmax or weighted average of DenseNet and SE-ResNeXt five-class CE softmax vectors
- Production ensemble weights: DenseNet `0.55`, SE-ResNeXt `0.45`, EfficientNet-B0 `0.00` (excluded)
- Heatmap: selected model's native CAM in single-model mode; per-case anatomy-gated native CAM in ensemble mode

`MODEL_MODE` accepts exactly `densenet121`, `se_resnext`, `efficientnet_b0`, or `ensemble`. Single-model modes load only their required checkpoint. Ensemble mode requires the DenseNet and SE-ResNeXt checkpoints; it does not load EfficientNet-B0. The application exits at startup if a required checkpoint is missing, incompatible, or declares the wrong architecture; it never falls back to random weights.

## Docker

Build and start with the checkpoint directory mounted read-only:

```bash
docker compose up --build -d
```

Equivalent direct command:

```bash
docker build -t knee-oa-native-cam:latest .
docker run --rm \
  --name knee-oa-native-cam \
  -p 8005:8005 \
  -e MODEL_MODE=ensemble \
  -v "$(pwd)/checkpoints:/app/checkpoints:ro" \
  knee-oa-native-cam:latest
```

For the production ensemble, use `-e MODEL_MODE=ensemble`. Its default voting
weights can be overridden with `ENSEMBLE_DENSENET_WEIGHT` and
`ENSEMBLE_SE_RESNEXT_WEIGHT`. `ENSEMBLE_EFFICIENTNET_B0_WEIGHT` is retained for
configuration compatibility but has no effect unless EfficientNet is added to a
future validated ensemble implementation.

Endpoints:

- Health: `GET http://127.0.0.1:8005/api/v1/health`
- Model information: `GET http://127.0.0.1:8005/api/v1/models`
- Prediction: `POST http://127.0.0.1:8005/api/v1/predict`
- ROI inspection: `POST http://127.0.0.1:8005/api/v1/predict/detect-roi`
- OpenAPI: `http://127.0.0.1:8005/docs`

Example:

```bash
curl -X POST http://127.0.0.1:8005/api/v1/predict \
  -F "file=@test_data/example.png"
```

Each detected knee returns its box, side, YOLO confidence, KL probabilities, ROI image, and `gradcam_image`. The historical `gradcam_image` field now contains the faithful native-CAM overlay, so existing clients keep the same JSON contract. The top-level response also includes the annotated source radiograph.

## Tests

```bash
pytest -q
```

The focused tests verify model-mode validation, strict checkpoint compatibility, probability-level soft voting, native-CAM/logit identity, per-case anatomy gating, heatmap dimensions, and laterality canonicalization.
