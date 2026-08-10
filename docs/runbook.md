# Local Runbook

## Start

1. Create `../env/ai.env` from `.env.example`.
2. Put the required checkpoints under `checkpoints/`.
3. Run:

```bash
make up
make ai-health
```

The service runs on `http://localhost:8005`.

## Common Commands

```bash
make status
make ai-logs
make test
make down
```

## Troubleshooting

| Problem | Check |
| --- | --- |
| API will not start | Check `make ai-logs` and checkpoint paths in `../env/ai.env`. |
| Port 8005 is busy | Stop the existing container with `make down`. |
| No knee is found | Use a frontal knee X-ray with the tibiofemoral joint visible. |
| Prediction looks wrong | Inspect the returned ROI and Grad-CAM. They are evidence, not a diagnosis. |

The container mounts `checkpoints/` as read-only. Do not place secrets or checkpoint files in Git.
