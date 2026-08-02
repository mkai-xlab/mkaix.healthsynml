import timm
import torch
import torch.nn as nn

from app.ml.models.base_model import BaseModel


class SEResNeXt50Model(BaseModel):
    """SE-ResNeXt-50 classifier with post-hoc Grad-CAM support."""

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

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Final spatial block used to generate post-hoc Grad-CAM."""
        return self.backbone.layer4

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
