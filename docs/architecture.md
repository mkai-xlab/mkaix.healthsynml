# Current Inference Architecture

The active local deployment is DenseNet-only. Older ensemble diagrams remain historical experiment artifacts and must not be treated as the current runtime.

```text
Uploaded frontal radiograph
        |
YOLOv8 joint detector
        |
1.15 x max-side centered square ROI
black padding only outside source image
        |
LAB CLAHE 1.25 -> square pad -> resize 384 -> ImageNet normalization
        |
timm DenseNet-121 (five CE logits)
        |
softmax KL grade 0-4 + predicted-class Grad-CAM at features.norm5
        |
stable JSON response: box, probabilities, ROI image, gradcam_image
```

The detector identifies the joint; it does not grade osteoarthritis. DenseNet grades the processed ROI. Grad-CAM visualizes class-discriminative evidence but is not segmentation or a clinical explanation guarantee.

Runtime choices are in `local.env`; the exact checkpoints and limitations are in [AI_REPORT.md](AI_REPORT.md).
