import torch
import torch.nn as nn
# import timm
from app.ml.models.base_model import BaseModel
import torchvision.models as models


class DenseNet121Model(BaseModel):
    """
    DenseNet-121 model wrapper subclass for Knee Osteoarthritis Kellgren-Lawrence Grade classification.
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(DenseNet121Model, self).__init__()

        # The 'pretrained' argument is deprecated. Use 'weights' instead.
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.model = models.densenet121(weights=weights)

        # Replace the classifier with a new one for our number of classes
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, num_classes)

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
    #     - Note: weights_only=True is used to safely deserialize weights without executing arbitrary code.
    #     """
    #     checkpoint = torch.load(path, map_location=device, weights_only=True)
    #     self.model.load_state_dict(checkpoint)
    #     self.model.eval()
    #     self.model.to(device)
    #     print(f"[Model] Successfully loaded weights from {path}")