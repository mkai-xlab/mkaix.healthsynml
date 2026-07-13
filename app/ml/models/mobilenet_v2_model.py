import torch
import torch.nn as nn
from app.ml.models.base_model import BaseModel
import torchvision.models as models

class MobileNetV2Model(BaseModel):
    """
    MobileNet-V2 model wrapper subclass for Knee Osteoarthritis Kellgren-Lawrence Grade classification.
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True, dropout_rate: float = 0.5):
        super(MobileNetV2Model, self).__init__()
        
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.model = models.mobilenet_v2(weights=weights)

        num_ftrs = self.model.classifier[1].in_features
        
        # Replace the classifier, increasing the dropout rate for better regularization
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=False), # Changed from p=0.2
            nn.Linear(num_ftrs, num_classes),
        )
        
        self.use_timm = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs standard forward pass
        """
        return self.model(x)
