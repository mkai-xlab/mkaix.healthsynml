import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.ml.models.base_model import BaseModel


class SEResNeXt50NativeCAMModel(BaseModel):
    """Inference-only SE-ResNeXt-50 with a five-map native-CAM head."""

    architecture = "final_native_cam_ce"

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
                "final_native_cam_ce requires CE logits; "
                f"received ordinal_type={ordinal_type!r}"
            )

        self.num_classes = num_classes
        self.backbone = timm.create_model(
            "seresnext50_32x4d",
            pretrained=pretrained,
            features_only=True,
            out_indices=(4,),
        )
        final_channels = self.backbone.feature_info.channels()[0]
        self.class_conv = nn.Conv2d(final_channels, num_classes, kernel_size=1)

    def class_maps(self, images: torch.Tensor) -> torch.Tensor:
        return self.class_conv(self.backbone(images)[0])

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
            raise ValueError("Native CAM currently expects one BxCxHxW inference sample")
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
