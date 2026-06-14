import torch
# import torch.nn as nn
# import timm
from app.ml.models.base_model import BaseModel
import torchvision.models as models

class EfficientNetB0Model(BaseModel):
    """
    EfficientNet-B0 model wrapper subclass for Knee Osteoarthritis Kellgren-Lawrence Grade classification.
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(EfficientNetB0Model, self).__init__()
        # Initialize the model using timm, setting the output classes to 5 (KL Grade 0 to 4)
        self.model = models.efficientnet_b0(pretrained=pretrained, num_classes=num_classes)
        # self.model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=num_classes)
        
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

