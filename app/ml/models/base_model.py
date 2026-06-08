import torch
import torch.nn as nn

class BaseModel(nn.Module):
    """
    Abstract Base Model Wrapper defining the contract for all neural networks
    in the Knee Osteoarthritis classification system.
    """
    def __init__(self):
        super(BaseModel, self).__init__()
        
    def load_weights(self, path: str, device: torch.device):
        """
        Loads weight checkpoint from disk onto the target device.
        """
        raise NotImplementedError("Subclasses must implement load_weights method")
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs forward pass and returns raw logits/probabilities.
        """
        raise NotImplementedError("Subclasses must implement predict method")
