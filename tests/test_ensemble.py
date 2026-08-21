"""
Tests for app.services.ensemble_service.

Purpose
-------
The ensemble service combines the predictions of multiple classification models
(DenseNet121 + SE-ResNeXt50) into a single output using a weighted soft-vote
scheme.  Additionally, when Grad-CAM visualisations are requested, it selects
the model whose confidence for the predicted class is highest so that the
CAM heatmap comes from the most decisive model.

Key concepts
------------
  - Logits vs probabilities: logits are raw model outputs (can be any real number);
    probabilities are the softmax of logits and always sum to 1.0.
  - Weighted soft vote: each model's softmax probabilities are multiplied by its
    weight, then the results are summed.  The weights should sum to 1.0.
  - Heatmap component selection: whichever model gives the highest probability
    for the predicted class is used as the CAM source, because a higher
    confidence means the model is more certain about that region.

Input
-----
  For weighted_soft_vote:
    logits    : dict {model_name: torch.Tensor (1, 5)}  — raw model outputs
    weights   : dict {model_name: float}               — weight for each model

  For select_heatmap_component:
    probabilities : dict {model_name: torch.Tensor (1, 5)}  — softmax probs
    predicted_class : int — the ensemble's predicted grade (0-4)

Expected output
---------------
  weighted_soft_vote:
      torch.Tensor (1, 5) — weighted average of softmax probabilities
      All values ∈ [0, 1]; sum along dim=1 == 1.0

  select_heatmap_component:
      str — the name of the model to use for the CAM heatmap
"""
import torch
import torch.nn.functional as F

from app.services.ensemble_service import ensemble_service


# ---------------------------------------------------------------------------
# Test: weighted_soft_vote normalises logits to softmax before averaging
# ---------------------------------------------------------------------------
def test_weighted_soft_vote_combines_probabilities_not_logits():
    """
    Input  :
      densenet logits   : [[2.0, 1.0, 0.0, -1.0, -2.0]]
      se_resnext logits : [[-1.0, 0.0, 3.0, 0.5, -0.5]]
      weights           : {densenet: 0.55, se_resnext: 0.45}

    How weighted_soft_vote should work (the correct way):
      1. Apply softmax to each model's logits → probabilities
      2. Multiply each model's probabilities by its weight
      3. Sum the weighted probabilities across models

    Expected output
      actual == expected (weighted average of softmax outputs)
      actual.sum(dim=1) == 1.0 (probabilities still sum to 1)
    """
    densenet_logits = torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0]])
    se_resnext_logits = torch.tensor([[-1.0, 0.0, 3.0, 0.5, -0.5]])

    logits = {
        "densenet121": densenet_logits,
        "seresnext50_32x4d": se_resnext_logits,
    }
    weights = {
        "densenet121": 0.55,
        "seresnext50_32x4d": 0.45,
    }
    actual = ensemble_service.weighted_soft_vote(logits, weights)

    # Compute expected: softmax then weighted average (the correct method)
    expected = (
        F.softmax(densenet_logits, dim=1) * 0.55
        + F.softmax(se_resnext_logits, dim=1) * 0.45
    )

    assert torch.allclose(actual, expected)

    # Softmax of weighted average must still sum to 1
    assert torch.allclose(actual.sum(dim=1), torch.ones(1))
    
    # This would be the wrong answer — proves we are NOT averaging logits first
    assert not torch.allclose(
        actual,
        F.softmax(
            (densenet_logits + se_resnext_logits) / 2.0,
            dim=1,
        ),
    )


# ---------------------------------------------------------------------------
# Test: weighted_soft_vote rejects single-model ensembles
# ---------------------------------------------------------------------------
def test_weighted_soft_vote_rejects_incompatible_inputs():
    """
    Input  : one model only: logits = {"only": zeros(1,5)}, weights = {"only": 1.0}

    Expected output
      ValueError whose message contains "at least two"
    """
    try:
        ensemble_service.weighted_soft_vote({"only": torch.zeros(1, 5)}, {"only": 1.0})
    except ValueError as error:
        assert "at least two" in str(error)
    else:
        raise AssertionError("Expected one-model voting to be rejected")


# ---------------------------------------------------------------------------
# Test: heatmap source is the model with highest confidence for predicted class
# ---------------------------------------------------------------------------
def test_heatmap_source_uses_highest_confidence_model():
    """
    Input  :
      probabilities:
        densenet   : [[0.05, 0.10, 0.55, 0.20, 0.10]]   # class-2: 0.55
        se_resnext : [[0.05, 0.10, 0.50, 0.25, 0.10]]   # class-2: 0.50
      predicted_class : 2

    Expected output
      "densenet121" — because 0.55 > 0.50 for class 2
    """
    probabilities = {
        "densenet121": torch.tensor([[0.05, 0.10, 0.55, 0.20, 0.10]]),
        "seresnext50_32x4d": torch.tensor([[0.05, 0.10, 0.50, 0.25, 0.10]]),
    }
    predicted_class = 2

    selected = ensemble_service.select_heatmap_component(
        probabilities, predicted_class
    )
    expected = max(
        probabilities,
        key=lambda name: probabilities[name][0, predicted_class].item(),
    )

    assert selected == expected


# ---------------------------------------------------------------------------
# Test: heatmap source falls back to confidence when models disagree
# ---------------------------------------------------------------------------
def test_heatmap_source_uses_confidence_when_models_disagree():
    """
    Input  :
      probabilities:
        densenet   : [[0.47, 0.20, 0.15, 0.10, 0.08]]   # class-0: 0.47
        se_resnext : [[0.39, 0.42, 0.10, 0.05, 0.04]]   # class-0: 0.39
      predicted_class : 0

    Expected output
      "densenet121" — densenet is more confident about class 0, so its
      activation map is the most meaningful for Grad-CAM.
    """
    probabilities = {
        "densenet121": torch.tensor([[0.47, 0.20, 0.15, 0.10, 0.08]]),
        "seresnext50_32x4d": torch.tensor([[0.39, 0.42, 0.10, 0.05, 0.04]]),
    }
    assert ensemble_service.select_heatmap_component(probabilities, 0) == "densenet121"
