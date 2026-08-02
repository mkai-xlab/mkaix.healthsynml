from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F


MIN_HEATMAP_JOINT_ENERGY = 0.55
MAX_HEATMAP_BORDER_ENERGY = 0.25
MAX_HEATMAP_LOWER_TIBIA_ENERGY = 0.25


class EnsembleService:
    """Combine classifier probabilities and select an ensemble Grad-CAM."""

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

        return sum(
            F.softmax(logits[name].float(), dim=1) * (weights[name] / total_weight)
            for name in logits
        )

    @staticmethod
    def select_heatmap_component(
        probabilities: Mapping[str, torch.Tensor],
        predicted_class: int,
        anatomy_metrics: Mapping[str, Mapping[str, float]],
    ) -> str:
        if set(probabilities) != set(anatomy_metrics):
            raise ValueError("Heatmap probabilities and anatomy metrics must align")

        def acceptable(name: str) -> bool:
            metrics = anatomy_metrics[name]
            return bool(
                metrics["joint_energy"] >= MIN_HEATMAP_JOINT_ENERGY
                and metrics["border_energy"] <= MAX_HEATMAP_BORDER_ENERGY
                and metrics["lower_tibia_energy"] <= MAX_HEATMAP_LOWER_TIBIA_ENERGY
                and metrics["peak_inside_joint"]
            )

        passing = [name for name in probabilities if acceptable(name)]
        candidates = passing or list(probabilities)
        return max(
            candidates,
            key=lambda name: (
                anatomy_metrics[name]["anatomy_score"]
                * float(probabilities[name][0, predicted_class].item()),
                anatomy_metrics[name]["anatomy_score"],
            ),
        )


ensemble_service = EnsembleService()
