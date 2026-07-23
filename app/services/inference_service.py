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

    def run_inference_with_class_maps(
        self, model: nn.Module, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one CNN pass and retain the maps that directly produce the logits."""
        if not hasattr(model, "forward_with_class_maps"):
            raise TypeError("The configured model does not expose native class maps")
        with torch.no_grad():
            return model.forward_with_class_maps(tensor)

# Singleton instance
inference_service = InferenceService()
