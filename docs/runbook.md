# Local Operations Runbook

## Start and Stop

```bash
make up
make status
make ai-health
make down
```

The API runs on `http://localhost:8005`; the response viewer runs on `http://localhost:8088`.

## Required Local Configuration

`local.env` must set `MODEL_MODE`, `MODEL_CHECKPOINT_PATH`, and `YOLO_CHECKPOINT_PATH`. Paths are relative to the repository and must resolve within the read-only `/app/checkpoints` mount.

The active local configuration is DenseNet-121 with the July 30 paired-view checkpoint and the July 26 YOLO detector. Verify the loaded architecture and epoch with `make ai-health` or the API health response.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| API does not start | `make ai-logs`; confirm both checkpoint paths in `local.env` exist. |
| Port is occupied | Stop the existing service with `make ai-down`, then run `make ai-up`. |
| No knee ROI | Upload a frontal knee radiograph with the complete tibiofemoral joint visible; the API returns a validation error when YOLO finds no joint. |
| Heatmap at ROI edge | Verify the app uses the 1.15 square ROI build. A heatmap is evidence visualization, not a diagnosis. |
| Viewer does not load | Run `make viewer-up`, then open `http://localhost:8088`. |

## Maintenance

Run `make experiments` only after archived report artifacts are updated. This refreshes the workbook and CSV inventory without training models or changing checkpoints.
