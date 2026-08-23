"""Shared ordinal-loss helpers for the Focal CORN pipeline notebooks.

These functions are written to be pasted verbatim into a notebook cell. They
implement Conditional Ordinal Regression for Normal-scores (CORN) with a focal
modulation on the per-task binary cross-entropy, following Shi et al. (2021).

Conventions:
    - num_classes = 5 (KL grades 0..4)
    - model emits num_classes - 1 = 4 binary logits, one per threshold k in [0, 3]
    - logits[k] represents P(y > k | x)
    - label_to_levels maps a class index to the level constraint vector
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


NUM_CLASSES = 5
NUM_TASKS = NUM_CLASSES - 1
TASK_WEIGHTS = (1.0, 1.2, 2.0, 3.5)
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
LABEL_SMOOTHING = 0.10


def label_to_levels(label: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Convert class indices to binary level vectors (CORN/CORAL convention).

    Example: class 2 with num_classes 5 -> [1, 1, 0, 0]
    """
    batch_size = label.size(0)
    levels = torch.zeros(batch_size, num_classes - 1, dtype=torch.float32, device=label.device)
    for i in range(batch_size):
        levels[i, : int(label[i].item())] = 1.0
    return levels


def corn_loss(
    logits: torch.Tensor,
    y_train: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    task_weights: tuple[float, ...] = TASK_WEIGHTS,
) -> torch.Tensor:
    """CORN loss: mean weighted BCE over K-1 binary threshold tasks."""
    loss = 0.0
    num_tasks = num_classes - 1
    for k in range(num_tasks):
        mask = y_train >= k
        if not mask.any():
            continue
        logits_k = logits[mask, k]
        targets_k = (y_train[mask] > k).float()
        targets_k = targets_k * (1 - LABEL_SMOOTHING) + (1 - targets_k) * LABEL_SMOOTHING
        w_k = task_weights[k] if k < len(task_weights) else 1.0
        loss = loss + w_k * F.binary_cross_entropy_with_logits(logits_k, targets_k)
    return loss / num_tasks


def focal_corn_loss(
    logits: torch.Tensor,
    y_train: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    gamma: float = FOCAL_GAMMA,
    alpha: float = FOCAL_ALPHA,
    task_weights: tuple[float, ...] = TASK_WEIGHTS,
) -> torch.Tensor:
    """Focal CORN loss: BCE + focal modulation (alpha * (1 - p_t) ** gamma)."""
    loss = 0.0
    num_tasks = num_classes - 1
    for k in range(num_tasks):
        mask = y_train >= k
        if not mask.any():
            continue
        logits_k = logits[mask, k]
        targets_k = (y_train[mask] > k).float()
        targets_k = targets_k * (1 - LABEL_SMOOTHING) + (1 - targets_k) * LABEL_SMOOTHING

        bce = F.binary_cross_entropy_with_logits(logits_k, targets_k, reduction="none")
        p = torch.sigmoid(logits_k)
        p_t = p * targets_k + (1 - p) * (1 - targets_k)
        focal_weight = alpha * (1 - p_t) ** gamma
        loss = loss + (focal_weight * bce).mean()
    return loss / num_tasks


def corn_probas(logits: torch.Tensor) -> torch.Tensor:
    """Convert K-1 binary logits to K-class probabilities via the chain rule.

    P(y = 0) = 1 - sigmoid(z_0)
    P(y = i) = prod_{k<i} sigmoid(z_k) * (1 - sigmoid(z_i)), for 0 < i < K-1
    P(y = K-1) = prod_{k<K-1} sigmoid(z_k)
    """
    cond_probas = torch.sigmoid(logits)
    batch_size = logits.size(0)
    num_classes = logits.size(1) + 1
    probas = torch.zeros(batch_size, num_classes, device=logits.device)
    cumprod = torch.cumprod(cond_probas, dim=1)
    probas[:, 0] = 1.0 - cond_probas[:, 0]
    for i in range(1, num_classes - 1):
        probas[:, i] = cumprod[:, i - 1] * (1.0 - cond_probas[:, i])
    probas[:, -1] = cumprod[:, -1]
    return probas


def corn_label_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Argmax over chain-rule probabilities (kept for parity with references)."""
    probas = corn_probas(logits)
    return torch.argmax(probas, dim=1)


__all__ = [
    "NUM_CLASSES",
    "NUM_TASKS",
    "FOCAL_GAMMA",
    "FOCAL_ALPHA",
    "label_to_levels",
    "corn_loss",
    "focal_corn_loss",
    "corn_probas",
    "corn_label_from_logits",
]