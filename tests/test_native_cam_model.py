from pathlib import Path

import torch

from app.ml.models.densenet121_model import DenseNet121Model
from app.ml.models.efficientnet_b0_model import EfficientNetB0Model
from app.ml.models.se_resnext50_32x4d_model import SEResNeXt50NativeCAMModel
from app.services.gradcam_service import NativeCAMService


CHECKPOINT = Path("checkpoints/densenet121/best_model.pth")
SE_RESNEXT_CHECKPOINT = Path(
    "checkpoints/se_resnext50_32x4d/best_model (1).pth"
)
EFFICIENTNET_B0_CHECKPOINT = Path("checkpoints/efficientnet_b0/best_model.pth")


def test_densenet_checkpoint_loads_and_generates_gradcam():
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    assert checkpoint["architecture"] == "timm_densenet121_linear_gradcam"

    model = DenseNet121Model(num_classes=5, pretrained=False, ordinal_type="ce")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    sample = torch.zeros(1, 3, 384, 384)
    with torch.no_grad():
        logits = model(sample)
    assert logits.shape == (1, 5)

    predicted_class = int(logits.argmax(dim=1).item())
    cam = NativeCAMService.extract_gradcam(
        model, sample, predicted_class, output_size=(384, 384)
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
    assert model.gradcam_target_layer is model.backbone.layer4

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


def test_efficientnet_b0_checkpoint_loads_and_native_cam_matches_logits():
    checkpoint = torch.load(
        EFFICIENTNET_B0_CHECKPOINT, map_location="cpu", weights_only=False
    )
    assert checkpoint["architecture"] == "efficientnet_b0_final_native_cam_ce"

    model = EfficientNetB0Model(
        num_classes=5, pretrained=False, ordinal_type="ce"
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    sample = torch.zeros(1, 3, 128, 128)
    with torch.no_grad():
        logits, class_maps = model.forward_with_class_maps(sample)

    assert class_maps.shape[:2] == (1, 5)
    assert logits.shape == (1, 5)
    assert torch.allclose(logits, class_maps.mean(dim=(2, 3)), atol=1e-7)

    predicted_class = int(logits.argmax(dim=1).item())
    cam = model.native_cam_from_class_maps(
        class_maps, predicted_class, output_size=(128, 128)
    )
    assert cam.shape == (128, 128)
    assert float(cam.min()) >= 0.0
    assert float(cam.max()) <= 1.0
