import torch
import torch.nn as nn
from app.ml.models.base_model import BaseModel
import torchvision.models as models

class EfficientNetB0Model(BaseModel):
    """
    EfficientNet-B0 model wrapper subclass for Knee Osteoarthritis Kellgren-Lawrence Grade classification.
    Implements a custom freeze_backbone method for partial fine-tuning.
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True, dropout_rate: float = 0.5):
        super(EfficientNetB0Model, self).__init__()
        
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = models.efficientnet_b0(weights=weights)

        num_ftrs = self.model.classifier[1].in_features
        # Increase dropout rate for better regularization
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(num_ftrs, num_classes),
        )
        
        self.use_timm = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs standard forward pass.
        """
        return self.model(x)

    def freeze_backbone(self):
        """
        Custom freezing strategy for EfficientNet-B0.
        Freezes the early blocks (0-3) and leaves the deeper blocks (4-7)
        and the classifier head trainable for fine-tuning.
        """
        print("Applying custom freezing strategy for EfficientNet-B0.")
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        for i in range(4, 8):
            print(f"  - Unfreezing feature block {i}...")
            for param in self.model.features[i].parameters():
                param.requires_grad = True

        print("  - Unfreezing classifier head...")
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """
        Unfreezes all parameters in the model for full fine-tuning.
        """
        print("Unfreezing all layers for full fine-tuning.")
        for param in self.model.parameters():
            param.requires_grad = True
