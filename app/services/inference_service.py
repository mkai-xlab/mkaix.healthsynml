import torch
import torch.nn as nn

class InferenceService:
    """Service to handle model inference forward passes."""
    def run_inference(self, model: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
        """
        Runs model forward pass inside no_grad context.
        """
        with torch.no_grad():
            logits = model(tensor)
        return logits

# Singleton instance
inference_service = InferenceService()
