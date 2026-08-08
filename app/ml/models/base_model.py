import torch.nn as nn


class BaseModel(nn.Module):
    """Minimal common base for classifiers loaded by the inference API."""

    def forward(self, images):  # pragma: no cover - implemented by subclasses
        """Define the classifier interface expected by the pipeline."""
        raise NotImplementedError
