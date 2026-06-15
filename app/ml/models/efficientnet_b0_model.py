import torch
import torch.nn as nn
# import timm
from app.ml.models.base_model import BaseModel
import torchvision.models as models

class EfficientNetB0Model(BaseModel):
    """
    EfficientNet-B0 model wrapper subclass for Knee Osteoarthritis Kellgren-Lawrence Grade classification.
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(EfficientNetB0Model, self).__init__()
        
        # The 'pretrained' argument is deprecated. Use 'weights' instead.
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = models.efficientnet_b0(weights=weights)

        # Replace the classifier with a new one for our number of classes
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_ftrs, num_classes),
        )
        
        # Add a flag for the base model to know this is not a timm model
        self.use_timm = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs standard forward pass
        """
        return self.model(x)
        
    # def load_weights(self, path: str, device: torch.device):
    #     """
    #     Loads trained weight checkpoints from the local disk or Drive
    #     """
    #     checkpoint = torch.load(path, map_location=device, weights_only=True)
    #     self.model.load_state_dict(checkpoint)
    #     self.model.eval()
    #     self.model.to(device)
    #     print(f"[Model] Successfully loaded weights from {path}")