"""
Tests for app.services.gradcam_service.GradCAMService.extract_gradcam().

Purpose
-------
This test module exercises extract_gradcam() without requiring real checkpoints
or the timm library.  A tiny synthetic model (_TinyGradCAMModel) replaces the
real classification backbone.

Input
-----
  - model      : any torch.nn.Module that has a .gradcam_target_layer property
                  returning a nn.Module whose output feature maps we want to analyse
  - image      : torch.Tensor of shape (1, 3, H, W) — a single image tensor
  - predicted_class : int — the class index that the model predicted for `image`
  - output_size    : tuple (H, W) — size to upsample the CAM to

Expected output
---------------
  numpy array of shape (H, W) where H, W are the output_size dimensions.
  All values are in the closed interval [0.0, 1.0].
"""
import numpy as np
import torch
import torch.nn as nn

from app.services.gradcam_service import GradCAMService


class _TinyGradCAMModel(nn.Module):
    """
    Minimal classification model that can run Grad-CAM without any external
    checkpoints or pretrained weights.

    Architecture (deliberately tiny so tests run fast):
      features  : Conv2d(3 → 4, kernel=3, padding=1)   — this is the gradcam target layer
      classifier: Linear(4 → 5, bias=False)             — 5 output classes (KL grades 0-4)

    The model does global average pooling between features and classifier,
    so the "feature maps" that Grad-CAM analyses are (4, H, W).
    """

    def __init__(self):
        super().__init__()
        # Single conv layer: 3 input channels (RGB) → 4 output channels
        self.features = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        # Final classifier: 4 pooled features → 5 KL grades
        self.classifier = nn.Linear(4, 5, bias=False)

    @property
    def gradcam_target_layer(self):
        # Grad-CAM hooks into the last convolutional layer.
        # For this tiny model, that's self.features.
        return self.features

    def forward(self, images):
        # ReLU is applied inside forward so that the conv output is non-negative.
        # Global average pooling collapses the spatial dimensions.
        features = torch.relu(self.features(images))
        pooled   = features.mean(dim=(2, 3))   # (batch, 4)
        return self.classifier(pooled)          # (batch, 5)


def test_extract_gradcam_returns_a_normalized_map_for_the_predicted_class():
    """
    Input  : a tiny model + a random 1×3×32×32 image, seeded for reproducibility

    What happens inside the test:
      1. model(image) → logits shape (1, 5)
      2. argmax(logits, dim=1) → predicted_class (any integer 0-4)
      3. GradCAMService.extract_gradcam(model, image, predicted_class, output_size=(32, 32))
         → returns a (32, 32) numpy array

    Expected output
      - shape    : (32, 32)  — matches output_size
      - all finite: no NaN or Inf values
      - range    : every element ∈ [0.0, 1.0]
    """
    torch.manual_seed(7)
    model = _TinyGradCAMModel().eval()
    image = torch.rand(1, 3, 32, 32)

    with torch.no_grad():
        predicted_class = int(model(image).argmax(dim=1).item())

    cam = GradCAMService.extract_gradcam(
        model,
        image,
        predicted_class=predicted_class,
        output_size=(32, 32),
    )

    assert cam.shape == (32, 32)
    assert np.isfinite(cam).all()
    assert 0.0 <= float(cam.min()) <= float(cam.max()) <= 1.0
