# Knee Osteoarthritis AI

FastAPI service that detects knee joints in an X-ray and predicts a Kellgren-Lawrence (KL) grade from 0 to 4 for each detected knee.

It is a research and capstone project. Predictions and Grad-CAM images are not clinical diagnoses.

## What It Does

1. YOLOv8 finds one or two knee joints.
2. Each joint is expanded into a square ROI so the joint margins remain visible.
3. A classifier predicts KL grades 0 to 4.
4. The API returns probabilities, the ROI, and a Grad-CAM heatmap.

The default model is DenseNet-121 with cross-entropy loss. Set `MODEL_MODE` to `densenet121`, `se_resnext`, or `ensemble` in the environment file.

## Run Locally

Create `../env/ai.env` from `.env.example`, then make sure the checkpoint paths exist.

```bash
make up
make ai-health
```

The API is available at `http://localhost:8005`. Interactive API docs are at `http://localhost:8005/docs`.

Useful commands:

```bash
make status
make ai-logs
make test
make down
```

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/v1/health` | Check that the service is running. |
| `GET /api/v1/models` | Show the active model and loss function. |
| `POST /api/v1/predict` | Detect knees and predict KL grades. |
| `POST /api/v1/predict/detect-roi` | Inspect detected knee ROIs without classification. |

Send PNG or JPEG files as the `file` form field.

## Project Docs

- [Architecture](docs/architecture.md)
- [Local runbook](docs/runbook.md)
- [CI and deployment](docs/ci_cd.md)
- [Unit tests](docs/testing.md)
- [Experiment notebooks](notebooks/experiments/README.md)
- [Paper reproduction notebooks](notebooks/paper/README.md)

Do not commit checkpoints, patient data, or environment files. Training records and figures belong in `docs/report/`.
