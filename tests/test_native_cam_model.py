from pathlib import Path

import torch

from app.ml.models.densenet121_model import DenseNet121Model
from app.ml.models.se_resnext50_32x4d_model import SEResNeXt50NativeCAMModel


CHECKPOINT = Path("checkpoints/densenet121/best_model.pth")
SE_RESNEXT_CHECKPOINT = Path(
    "checkpoints/se_resnext50_32x4d/best_model (1).pth"
)


def test_checkpoint_loads_and_native_cam_matches_logits():
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    assert checkpoint["architecture"] == "canonical_final_linear_cam"

    model = DenseNet121Model(num_classes=5, pretrained=False, ordinal_type="ce")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    sample = torch.zeros(1, 3, 384, 384)
    with torch.no_grad():
        logits, class_maps = model.forward_with_class_maps(sample)

    assert class_maps.shape == (1, 5, 12, 12)
    assert logits.shape == (1, 5)
    assert torch.allclose(logits, class_maps.mean(dim=(2, 3)), atol=1e-7)

    predicted_class = int(logits.argmax(dim=1).item())
    cam = model.native_cam_from_class_maps(
        class_maps, predicted_class, output_size=(384, 384)
    )
    assert cam.shape == (384, 384)
    assert float(cam.min()) >= 0.0
    assert float(cam.max()) <= 1.0


def test_se_resnext_checkpoint_loads_and_native_cam_matches_logits():
    checkpoint = torch.load(
        SE_RESNEXT_CHECKPOINT, map_location="cpu", weights_only=False
    )
    assert checkpoint["architecture"] == "final_native_cam_ce"

    model = SEResNeXt50NativeCAMModel(
        num_classes=5, pretrained=False, ordinal_type="ce"
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    sample = torch.zeros(1, 3, 384, 384)
    with torch.no_grad():
        logits, class_maps = model.forward_with_class_maps(sample)

    assert class_maps.shape == (1, 5, 12, 12)
    assert logits.shape == (1, 5)
    assert torch.allclose(logits, class_maps.mean(dim=(2, 3)), atol=1e-7)

    predicted_class = int(logits.argmax(dim=1).item())
    cam = model.native_cam_from_class_maps(
        class_maps, predicted_class, output_size=(384, 384)
    )
    assert cam.shape == (384, 384)
    assert float(cam.min()) >= 0.0
    assert float(cam.max()) <= 1.0
