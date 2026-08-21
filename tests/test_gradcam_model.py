"""
Tests for app.ml.models.densenet121_model and
app.ml.models.se_resnext50_32x4d_model.

Purpose
-------
These tests load real trained checkpoints (when available) and verify:
  1. The checkpoint metadata (architecture key) matches expectations.
  2. The model can be restored from the saved state_dict.
  3. The restored model can do a forward pass.
  4. Grad-CAM activation maps can be extracted for the model.

These are integration tests — they require actual checkpoint files to be
present in the ml/checkpoints/ directory (typically mounted from Google Drive).
If a checkpoint is missing, pytest.skip() is called so the test does not fail.

Input
-----
  - Checkpoint file path, e.g. checkpoints/densenet121/best_model.pth
    The file must be a dict saved with torch.save() containing at least:
      {
        "architecture": str,          # e.g. "timm_densenet121_linear_gradcam"
        "model_state_dict": dict,    # torch state_dict of the model
      }

Expected output
---------------
  - checkpoint["architecture"] == expected architecture string
  - model.load_state_dict(...) succeeds with strict=True
  - model forward pass on a zero tensor of shape (1, 3, 384, 384) succeeds
  - GradCAMService.extract_gradcam(model, ...) returns a (384, 384) map
    with all values in [0.0, 1.0]
"""
from pathlib import Path

import pytest
import torch

from app.ml.models.densenet121_model import DenseNet121Model
from app.ml.models.se_resnext50_32x4d_model import SEResNeXt50Model
from app.services.gradcam_service import GradCAMService


# ---------------------------------------------------------------------------
# Paths to real trained checkpoints (mounted from Google Drive)
# ---------------------------------------------------------------------------
CHECKPOINT = Path("checkpoints/densenet121/best_model.pth")
SE_RESNEXT_CHECKPOINT = Path(
    "checkpoints/se_resnext50_32x4d/"
    "2026-08-08_08-35-38_UTC_linear_gradcam/best_model.pth"
)


# ---------------------------------------------------------------------------
# Helper: load a checkpoint, skipping the test if the file is absent.
# ---------------------------------------------------------------------------
# Input  : Path to a .pth checkpoint file
# Output : the deserialised dict returned by torch.load(...)
#          or pytest.skip() if the file does not exist
def _load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        # Checkpoint not mounted — skip this test without failure
        pytest.skip(f"Optional checkpoint is not mounted: {path}")
    # weights_only=False: checkpoint contains non-tensor objects (e.g. strings)
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Helper: run Grad-CAM on a model and verify basic shape/value constraints.
# ---------------------------------------------------------------------------
# Input  : model (torch.nn.Module), sample input tensor (1, 3, 384, 384)
# Output : raises AssertionError if any check fails
def _assert_gradcam(model: torch.nn.Module, sample: torch.Tensor) -> None:
    with torch.no_grad():
        # Run forward pass to get the predicted class
        predicted_class = int(model(sample).argmax(dim=1).item())
    # Extract Grad-CAM for the predicted class
    cam = GradCAMService.extract_gradcam(
        model, sample, predicted_class, output_size=tuple(sample.shape[-2:])
    )
    # Shape must match the spatial dimensions of the input tensor
    # for image_size = 384, the shape should be (384, 384)
    assert cam.shape == tuple(sample.shape[-2:])
    # Normalised CAM values are always in [0, 1]
    assert float(cam.min()) >= 0.0
    assert float(cam.max()) <= 1.0


# ---------------------------------------------------------------------------
# Test: DenseNet121 checkpoint
# ---------------------------------------------------------------------------
def test_densenet_checkpoint_loads_and_generates_gradcam():
    """
    Input  : checkpoints/densenet121/best_model.pth
    Expected output
      - checkpoint architecture tag : "timm_densenet121_linear_gradcam"
      - model loads successfully from state_dict
      - Grad-CAM on a zero input produces a valid (384, 384) map
    """
    checkpoint = _load_checkpoint(CHECKPOINT)
    assert checkpoint["architecture"] == "timm_densenet121_linear_gradcam"

    model = DenseNet121Model(num_classes=5, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    _assert_gradcam(model, torch.zeros(1, 3, 384, 384))


# ---------------------------------------------------------------------------
# Test: SE-ResNeXt50 checkpoint
# ---------------------------------------------------------------------------
def test_se_resnext_checkpoint_loads_and_generates_gradcam():
    """
    Input  : checkpoints/se_resnext50_32x4d/.../best_model.pth
    Expected output
      - checkpoint architecture tag : "seresnext50_32x4d_linear_gradcam"
      - model loads successfully from state_dict
      - Grad-CAM on a zero input produces a valid (384, 384) map
    """
    checkpoint = _load_checkpoint(SE_RESNEXT_CHECKPOINT)
    assert checkpoint["architecture"] == "seresnext50_32x4d_linear_gradcam"

    model = SEResNeXt50Model(num_classes=5, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    _assert_gradcam(model, torch.zeros(1, 3, 384, 384))
