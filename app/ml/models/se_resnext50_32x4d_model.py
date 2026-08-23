import timm
import torch
import torch.nn as nn
from app.ml.models.base_model import BaseModel


class SEResNeXt50Model(BaseModel):
    """SE-ResNeXt-50 classifier with post-hoc Grad-CAM support."""

    architecture = "seresnext50_32x4d_linear_gradcam"

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model(
            "seresnext50_32x4d",
            pretrained=pretrained,

            # do not include the final classification head, because we want to use the features for Grad-CAM
            features_only=True,

            # use the final spatial block (layer4) for Grad-CAM
            out_indices=(4,),
        )

        # use the final feature map channels to create a new classification head
        final_channels = self.backbone.feature_info.channels()[0]
        self.classifier = nn.Linear(final_channels, num_classes)

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Final spatial block used to generate post-hoc Grad-CAM."""

        # model has 4 stage -> choose the last stage for Grad-CAM
        return self.backbone.layer4

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)[0]

        # average the features over the spatial dimensions to get a single vector -> Global Average Pooling
        # because the features are 3D (B, C, H, W) -> (B, C, 1, 1)
        pooled_features = features.mean(dim=(2, 3))

        return self.classifier(pooled_features)
