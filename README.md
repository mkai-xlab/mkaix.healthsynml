# Knee Osteoarthritis Native-CAM API

FastAPI inference service for Kellgren-Lawrence grades 0-4. The production path combines YOLOv8 knee-joint detection with the `canonical_final_linear_cam` DenseNet-121 checkpoint.

The classifier is inference-only. A 1x1 convolution creates five grade-specific maps and global spatial averaging creates the five CE logits. The returned native CAM is therefore the actual predicted-grade map, not a hook-based Grad-CAM approximation.

## Runtime Contract

- DenseNet checkpoint: `checkpoints/densenet121/best_model.pth`
- YOLO checkpoint: `checkpoints/yolov8/best.pt`
- Preprocessing: laterality canonicalization, square padding, CLAHE, resize to 400, center crop to 384, ImageNet normalization
- Right knee convention: right ROIs are horizontally mirrored before classification
- Classification: five CE logits with softmax probabilities
- Heatmap: positive predicted-class native CAM, upsampled from 12x12 to the processed 384x384 ROI

The application exits at startup if the DenseNet checkpoint is missing, incompatible, or does not declare `canonical_final_linear_cam`. It never falls back to random weights.

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

The focused tests verify strict checkpoint compatibility, the defining native-CAM/logit identity, heatmap dimensions, and laterality canonicalization.
