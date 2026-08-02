# Knee Osteoarthritis AI

FastAPI inference service for Kellgren-Lawrence grades 0-4. The current local production mode combines YOLOv8 knee-joint detection with a DenseNet-121 classifier and Grad-CAM explanation.

## Current Production Setup

- Model mode: `densenet121`
- Classifier: `timm_densenet121_linear_gradcam`, CE, epoch 4
- Classifier checkpoint: `checkpoints/densenet121/2026-07-30_09-03-29_850983_UTC_paired_view_yolo_roi/best_model.pth`
- YOLO checkpoint: `checkpoints/yolov8/2026-07-26_20-49-25_joint_detection/best.pt`
- ROI: YOLO box expanded by `1.15 x` its largest dimension, centered square, black padding only outside the source image
- Preprocessing: LAB CLAHE 1.25, square pad, resize `384 x 384`, ImageNet normalization
- Explanation: predicted-class Grad-CAM from DenseNet `features.norm5`

The API returns the historical `gradcam_image` field. It is a Grad-CAM overlay when `MODEL_MODE=densenet121`.

## Start Locally

`local.env` selects the local model and detector paths. Do not commit checkpoint files or private environment files.

```bash
make up
make ai-health
```

Open API documentation at `http://localhost:8005/docs`. Open the prediction-response viewer at `http://localhost:8088` and paste a complete `/api/v1/predict` response.

Useful targets:

```bash
make status
make ai-logs
make viewer-logs
make test
make experiments
make down
```

`make experiments` rebuilds [all_experiments.xlsx](docs/report/all_experiments.xlsx) and the CSV tabs in `docs/report/summary/` from archived report artifacts.

## API Contract

- `GET /api/v1/health`
- `GET /api/v1/models`
- `POST /api/v1/predict` with form field `file`
- `POST /api/v1/predict/detect-roi` with form field `file`

Each detected knee contains the raw YOLO box, side, detector confidence, grade probabilities, ROI image, and heatmap. The output schema is stable; do not add model-specific fields to it without a compatibility decision.

## Training and Reports

- Controlled notebooks: [notebooks/experiments](notebooks/experiments/README.md)
- DenseNet history: [report.md](docs/report/dense_net_121/report.md)
- Current production record: [AI_REPORT.md](docs/AI_REPORT.md)
- Current runtime architecture: [architecture.md](docs/architecture.md)
- Experiment inventory: [all_experiments.xlsx](docs/report/all_experiments.xlsx)

Historical notebooks are retained for provenance. Completed runs must have an exact timestamp, checkpoint path, metrics, and visual evidence before being considered for promotion.

## Limitations

KL grade is a whole-knee label. Grad-CAM is not a segmentation mask and cannot prove that a prediction is clinically correct. The system is for research and capstone demonstration, not independent clinical diagnosis.
