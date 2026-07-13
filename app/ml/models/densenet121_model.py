import torch
import torch.nn as nn
from app.ml.models.base_model import BaseModel
import torchvision.models as models

class DenseNet121Model(BaseModel):
    """
    DenseNet-121 model wrapper subclass for Knee Osteoarthritis Kellgren-Lawrence Grade classification.
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True, dropout_rate: float = 0.5):
        super(DenseNet121Model, self).__init__()
        
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.model = models.densenet121(weights=weights)

        num_ftrs = self.model.classifier.in_features
        # Add a Dropout layer before the final linear layer for regularization
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_ftrs, num_classes)
        )
        
        self.use_timm = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs standard forward pass
        """
        return self.model(x)
