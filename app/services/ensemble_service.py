from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F


class EnsembleService:
    """Soft-vote probabilities and choose one component's Grad-CAM per case."""

    @staticmethod
    def weighted_soft_vote( logits: Mapping[str, torch.Tensor], weights: Mapping[str, float]) -> torch.Tensor:
        """"Soft-vote calibrated class probabilities from multiple models.
        Args:
            logits: A mapping of model names to their output logits tensors.
            weights: A mapping of model names to their corresponding weights for voting.
        Returns:
            A tensor representing the soft-voted class probabilities.
        Raises:
            ValueError: If the input logits or weights are invalid.
        """

        # case models counts < 2
        if len(logits) < 2:
            raise ValueError("Soft voting requires at least two logits tensors")

        # case logis & weights has different model names or different shapes 
        if set(logits) != set(weights):
            raise ValueError("Ensemble logits and weights must have identical model names")


        # list of logits values to check shapes ([n, 1, 5])
        values = list(logits.values())


        # case logits have different shapes or weights are not finite and non-negative
        if any(value.shape != values[0].shape for value in values[1:]):
            raise ValueError("Ensemble logits must have identical shapes")

        # case weights are not finite and non-negative
        if any(not np.isfinite(weight) or weight < 0 for weight in weights.values()):
            raise ValueError("Ensemble weights must be finite and non-negative")

        # case weights sum to zero
        total_weight = float(sum(weights.values()))
        if total_weight <= 0:
            raise ValueError("At least one ensemble weight must be positive")

        # Vote on calibrated class probabilities, not raw logits from different backbones.
        return sum(
            F.softmax(logits[name].float(), dim=1) * (weights[name] / total_weight)
            for name in logits
        )

    @staticmethod
    def select_heatmap_component( probabilities: Mapping[str, torch.Tensor],predicted_class: int,) -> str:

        """Select the model with the highest probability for the predicted class.
        
        Args:
            probabilities: A mapping of model names to their output probability tensors.
            predicted_class: The index of the predicted class.
        Returns:
            The name of the model with the highest probability for the predicted class.(densenet121, efficientnet_b0, resnet50, vit_b_16, etc)

        """
        if not probabilities:
            raise ValueError("At least one model probability tensor is required")

        # return the model with the highest probability for the predicted class
        return max(
            probabilities,
            key=lambda name: float(probabilities[name][0, predicted_class].item()),
        )


ensemble_service = EnsembleService()
