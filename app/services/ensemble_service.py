from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F


class EnsembleService:
    """Soft-vote probabilities and choose one component's Grad-CAM per case."""

    @staticmethod
    def weighted_soft_vote(
        logits: Mapping[str, torch.Tensor], weights: Mapping[str, float]
    ) -> torch.Tensor:
        if len(logits) < 2:
            raise ValueError("Soft voting requires at least two logits tensors")
        if set(logits) != set(weights):
            raise ValueError("Ensemble logits and weights must have identical model names")
        values = list(logits.values())
        if any(value.shape != values[0].shape for value in values[1:]):
            raise ValueError("Ensemble logits must have identical shapes")
        if any(not np.isfinite(weight) or weight < 0 for weight in weights.values()):
            raise ValueError("Ensemble weights must be finite and non-negative")
        total_weight = float(sum(weights.values()))
        if total_weight <= 0:
            raise ValueError("At least one ensemble weight must be positive")

        # Vote on calibrated class probabilities, not raw logits from different backbones.
        return sum(
            F.softmax(logits[name].float(), dim=1) * (weights[name] / total_weight)
            for name in logits
        )

    @staticmethod
    def select_heatmap_component(
        probabilities: Mapping[str, torch.Tensor],
        predicted_class: int,
    ) -> str:
        if not probabilities:
            raise ValueError("At least one model probability tensor is required")
        return max(
            probabilities,
            key=lambda name: float(probabilities[name][0, predicted_class].item()),
        )


ensemble_service = EnsembleService()
