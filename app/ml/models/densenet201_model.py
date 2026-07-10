import torch
import torch.nn as nn
from app.ml.models.base_model import BaseModel
import timm

class DenseNet201Model(BaseModel):
    """
    DenseNet-201 model wrapper subclass for Knee Osteoarthritis Kellgren-Lawrence Grade classification.
    Supports standard classification, CORAL (threshold), and CORN (conditional ordinal).
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True, dropout_rate: float = 0.5, ordinal_type: str = "threshold"):
        super(DenseNet201Model, self).__init__()
        
        # Determine output features based on ordinal type
        # For CORAL ("threshold") and CORN ("corn" / "focal_corn"), output features = num_classes - 1 = 4
        # For expected value or standard classification, output features = num_classes = 5
        out_features = num_classes - 1 if ordinal_type in ["threshold", "corn", "focal_corn"] else num_classes
        
        # Load standard DenseNet-201 model using timm
        self.model = timm.create_model('densenet201', pretrained=pretrained, num_classes=out_features, drop_rate=dropout_rate)
        
        self.use_timm = True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs standard forward pass
        """
        return self.model(x)
