import numpy as np
import torch
import torch.nn as nn

from app.services.gradcam_service import GradCAMService


class _TinyGradCAMModel(nn.Module):
    """Small model that exercises Grad-CAM without checkpoints or timm."""

    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        self.classifier = nn.Linear(4, 5, bias=False)

    @property
    def gradcam_target_layer(self):
        return self.features

    def forward(self, images):
        features = torch.relu(self.features(images))
        return self.classifier(features.mean(dim=(2, 3)))


def test_extract_gradcam_returns_a_normalized_map_for_the_predicted_class():
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
