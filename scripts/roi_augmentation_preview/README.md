# Knee ROI Augmentation Preview

Build and run from the repository root:

```bash
docker build -t knee-roi-augmentation-preview:latest scripts/roi_augmentation_preview
docker run --rm -d --name knee-roi-augmentation-preview -p 8090:8090 knee-roi-augmentation-preview:latest
```

Open `http://127.0.0.1:8090`, or request an image directly:

```bash
curl -F "image=@roi.png" "http://127.0.0.1:8090/augment?seed=42" -o augmented.png
```

The output is a PNG rather than base64. The `X-Augmentation-Seed` and
`X-Augmentation-Operations` response headers record the sampled transformation.
