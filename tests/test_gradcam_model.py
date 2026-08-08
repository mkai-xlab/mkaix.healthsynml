from pathlib import Path

import pytest
import torch

from app.ml.models.densenet121_model import DenseNet121Model
from app.ml.models.se_resnext50_32x4d_model import SEResNeXt50Model
from app.services.gradcam_service import GradCAMService


CHECKPOINT = Path("checkpoints/densenet121/best_model.pth")
SE_RESNEXT_CHECKPOINT = Path(
    "checkpoints/se_resnext50_32x4d/"
    "2026-08-08_08-35-38_UTC_linear_gradcam/best_model.pth"
)
def _load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        pytest.skip(f"Optional checkpoint is not mounted: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _assert_gradcam(model: torch.nn.Module, sample: torch.Tensor) -> None:
    with torch.no_grad():
        predicted_class = int(model(sample).argmax(dim=1).item())
    cam = GradCAMService.extract_gradcam(
        model, sample, predicted_class, output_size=tuple(sample.shape[-2:])
    )
    assert cam.shape == tuple(sample.shape[-2:])
    assert float(cam.min()) >= 0.0
    assert float(cam.max()) <= 1.0


def test_densenet_checkpoint_loads_and_generates_gradcam():
    checkpoint = _load_checkpoint(CHECKPOINT)
    assert checkpoint["architecture"] == "timm_densenet121_linear_gradcam"
    model = DenseNet121Model(num_classes=5, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    _assert_gradcam(model, torch.zeros(1, 3, 384, 384))


def test_se_resnext_checkpoint_loads_and_generates_gradcam():
    checkpoint = _load_checkpoint(SE_RESNEXT_CHECKPOINT)
    assert checkpoint["architecture"] == "seresnext50_32x4d_linear_gradcam"
    model = SEResNeXt50Model(num_classes=5, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    _assert_gradcam(model, torch.zeros(1, 3, 384, 384))
