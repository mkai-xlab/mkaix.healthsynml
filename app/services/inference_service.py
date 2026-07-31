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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return logits and optional native maps without building a graph."""
        with torch.no_grad():
            if hasattr(model, "forward_with_class_maps"):
                return model.forward_with_class_maps(tensor)
            return model(tensor), None

# Singleton instance
inference_service = InferenceService()
