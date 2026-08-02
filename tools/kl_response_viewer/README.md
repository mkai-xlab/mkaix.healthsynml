# KL Response Viewer

The viewer is embedded in the FastAPI image at `http://localhost:8005/result-viewer`.
It renders the existing `POST /api/v1/predict` JSON response in the browser. Image
fields are decoded locally and are never uploaded or sent to another server.

```bash
make up
```

Open `http://localhost:8005/result-viewer`, paste the complete response, and select **Render
response**.
