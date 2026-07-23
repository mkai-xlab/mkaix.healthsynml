# Knee Osteoarthritis Soft-Voting Native-CAM API

FastAPI inference service for Kellgren-Lawrence grades 0-4. The inference path combines YOLOv8 knee-joint detection with an equal soft-voting ensemble of DenseNet-121 and SE-ResNeXt-50.

Both classifiers are inference-only native-CAM models. Their five-class softmax probability vectors are averaged with equal weight. The response heatmap uses the SE-ResNeXt class map for the ensemble-selected grade because the completed audit showed higher joint energy and lower border energy than DenseNet.

## Runtime Contract

- DenseNet checkpoint: `checkpoints/densenet121/best_model.pth`
- SE-ResNeXt checkpoint: `checkpoints/se_resnext50_32x4d/best_model (1).pth`
- YOLO checkpoint: `checkpoints/yolov8/best.pt`
- Preprocessing: laterality canonicalization, square padding, CLAHE, resize to 400, center crop to 384, ImageNet normalization
- Right knee convention: right ROIs are horizontally mirrored before classification
- Classification: equal average of the two five-class CE softmax vectors
- Heatmap: positive SE-ResNeXt native CAM for the ensemble-selected grade, upsampled from 12x12 to the processed 384x384 ROI

The application exits at startup if either checkpoint is missing, incompatible, or declares the wrong architecture. It never falls back to random weights or a one-model prediction.

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
  -v "$(pwd)/checkpoints:/app/checkpoints:ro" \
  knee-oa-native-cam:latest
```

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

The focused tests verify strict compatibility for both checkpoints, equal probability-level soft voting, each native-CAM/logit identity, heatmap dimensions, and laterality canonicalization.
