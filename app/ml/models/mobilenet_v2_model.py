import torch
import torch.nn as nn
from app.ml.models.base_model import BaseModel
import torchvision.models as models

class MobileNetV2Model(BaseModel):
    """
    MobileNet-V2 model wrapper subclass for Knee Osteoarthritis Kellgren-Lawrence Grade classification.
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(MobileNetV2Model, self).__init__()
        
        # Use the modern 'weights' API instead of 'pretrained'
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.model = models.mobilenet_v2(weights=weights)

        # In MobileNetV2, the classifier is a Sequential block: (0): Dropout, (1): Linear
        # Get the number of input features for the Linear layer
        num_ftrs = self.model.classifier[1].in_features
        
        # Replace the classifier with a new one matching our number of classes
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(num_ftrs, num_classes),
        )
        
        # Add a flag for the base model to know this is not a timm model
        self.use_timm = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs standard forward pass
        """
        return self.model(x)