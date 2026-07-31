# KL Response Viewer

Standalone browser viewer for the existing `POST /predict` JSON response. Image
fields are decoded in the browser and are never uploaded or sent to another server.

```bash
docker build -t kl-response-viewer:latest tools/kl_response_viewer
docker run --rm -p 8088:8080 --name kl-response-viewer kl-response-viewer:latest
```

Open `http://localhost:8088`, paste the complete response, and select **Render
response**.
