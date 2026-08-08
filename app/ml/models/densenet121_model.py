import torch
import torch.nn as nn
import timm

from app.ml.models.base_model import BaseModel


class DenseNet121Model(BaseModel):
    """Standard five-logit DenseNet-121 with post-hoc Grad-CAM support."""

    architecture = "timm_densenet121_linear_gradcam"

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model(
            "densenet121",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=0.20,
        )

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Final spatial DenseNet feature tensor used by the training audit."""
        return self.backbone.features.norm5

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)
