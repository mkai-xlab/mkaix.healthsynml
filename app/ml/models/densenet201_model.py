import torch
import torch.nn as nn
from app.ml.models.base_model import BaseModel
import timm

class DenseNet201Model(BaseModel):
    """
    DenseNet-201 model wrapper subclass for Knee Osteoarthritis Kellgren-Lawrence Grade classification.
    Supports standard classification, CORAL (threshold), and CORN (conditional ordinal).
    Uses SOTA Multi-Scale Feature Extractor (out_indices 2, 3, 4 represent transitions and final block).
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True, dropout_rate: float = 0.5, ordinal_type: str = "threshold"):
        super(DenseNet201Model, self).__init__()
        
        # Determine output features based on ordinal type
        # For CORAL ("threshold") and CORN ("corn" / "focal_corn"), output features = num_classes - 1 = 4
        # For expected value or standard classification, output features = num_classes = 5
        out_features = num_classes - 1 if ordinal_type in ["threshold", "corn", "focal_corn"] else num_classes
        
        # SOTA Multi-Scale Feature Extractor (out_indices 2, 3, 4 represent transitions and final block)
        self.backbone = timm.create_model(
            'densenet201', 
            pretrained=pretrained, 
            features_only=True, 
            out_indices=(2, 3, 4)
        )
        
        # timm densenet201 stage channels: stage 2 = 512, stage 3 = 1792, stage 4 = 1920
        total_channels = 512 + 1792 + 1920
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(total_channels, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, out_features)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled = [self.gap(f).flatten(1) for f in features]
        x = torch.cat(pooled, dim=1)
        return self.classifier(x)

    def freeze_backbone(self):
        print("Freezing DenseNet-201 backbone features.")
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        print("Unfreezing all DenseNet-201 parameters.")
        for param in self.parameters():
            param.requires_grad = True

    def unfreeze_last_block(self):
        print("Unfreezing only last dense block (denseblock4) and classifier head.")
        for param in self.parameters():
            param.requires_grad = False
        features = getattr(self.backbone, "features", self.backbone)
        if hasattr(features, "denseblock4"):
            for param in features.denseblock4.parameters():
                param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
