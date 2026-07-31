import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from app.ml.models.base_model import BaseModel


class EfficientNetB0Model(BaseModel):
    """Inference-only EfficientNet-B0 with the trained five-map CAM head."""

    architecture = "efficientnet_b0_final_native_cam_ce"

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = False,
        ordinal_type: str = "ce",
        **_: object,
    ):
        super().__init__()
        if ordinal_type != "ce":
            raise ValueError(
                "efficientnet_b0_final_native_cam_ce requires CE logits; "
                f"received ordinal_type={ordinal_type!r}"
            )

        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        network = models.efficientnet_b0(weights=weights)
        self.features = network.features
        self.class_conv = nn.Conv2d(
            network.classifier[1].in_features, num_classes, kernel_size=1
        )

    def class_maps(self, images: torch.Tensor) -> torch.Tensor:
        return self.class_conv(self.features(images))

    @staticmethod
    def logits_from_class_maps(class_maps: torch.Tensor) -> torch.Tensor:
        return class_maps.mean(dim=(2, 3))

    def forward_with_class_maps(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        class_maps = self.class_maps(images)
        return self.logits_from_class_maps(class_maps), class_maps

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_class_maps(images)
        return logits

    @staticmethod
    def native_cam_from_class_maps(
        class_maps: torch.Tensor,
        class_index: int,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        if class_maps.ndim != 4 or class_maps.size(0) != 1:
            raise ValueError("Native CAM currently expects one BxCxHxW sample")
        if not 0 <= class_index < class_maps.size(1):
            raise ValueError(f"Class index is out of range: {class_index}")

        cam = F.relu(class_maps[:, class_index : class_index + 1])
        cam = F.interpolate(
            cam,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return cam / cam.max().clamp_min(1e-8)
