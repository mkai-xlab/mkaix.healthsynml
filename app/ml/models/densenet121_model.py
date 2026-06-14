import torch
# import timm
from app.ml.models.base_model import BaseModel
import torchvision.models as models


class DenseNet121Model(BaseModel):
    """
    DenseNet-121 model wrapper subclass for Knee Osteoarthritis Kellgren-Lawrence Grade classification.
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(DenseNet121Model, self).__init__()
        # Initialize DenseNet-121 using timm, mapping to 5 output classes (KL Grade 0 to 4)
        # self.model = timm.create_model('densenet121', pretrained=pretrained, num_classes=num_classes)
        self.model = models.densenet121(pretrained=pretrained,num_classes= num_classes)
        
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

