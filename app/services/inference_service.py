import torch
import torch.nn as nn


class InferenceService:
    """Run classification logits without constructing a PyTorch gradient graph."""

    def run_inference(self, model: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
        """Run the normal prediction pass; Grad-CAM uses a separate gradient pass."""
        with torch.no_grad():
            return model(tensor)


# Shared stateless helper for the normal (non-explanation) forward pass.
inference_service = InferenceService()
