import torch
import torch.nn.functional as F

from app.services.ensemble_service import ensemble_service


def test_weighted_soft_vote_combines_probabilities_not_logits():
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
    expected = (
        F.softmax(densenet_logits, dim=1) * 0.55
        + F.softmax(se_resnext_logits, dim=1) * 0.45
    )

    assert torch.allclose(actual, expected)
    assert torch.allclose(actual.sum(dim=1), torch.ones(1))
    assert not torch.allclose(
        actual,
        F.softmax(
            (densenet_logits + se_resnext_logits) / 2.0,
            dim=1,
        ),
    )


def test_weighted_soft_vote_rejects_incompatible_inputs():
    try:
        ensemble_service.weighted_soft_vote({"only": torch.zeros(1, 5)}, {"only": 1.0})
    except ValueError as error:
        assert "at least two" in str(error)
    else:
        raise AssertionError("Expected one-model voting to be rejected")


def test_heatmap_source_uses_highest_confidence_model():
    probabilities = {
        "densenet121": torch.tensor([[0.05, 0.10, 0.55, 0.20, 0.10]]),
        "seresnext50_32x4d": torch.tensor([[0.05, 0.10, 0.50, 0.25, 0.10]]),
    }
    selected = ensemble_service.select_heatmap_component(probabilities, 2)

    assert selected == "seresnext50_32x4d"


def test_heatmap_source_uses_confidence_when_models_disagree():
    probabilities = {
        "densenet121": torch.tensor([[0.47, 0.20, 0.15, 0.10, 0.08]]),
        "seresnext50_32x4d": torch.tensor([[0.39, 0.42, 0.10, 0.05, 0.04]]),
    }
    assert ensemble_service.select_heatmap_component(probabilities, 0) == "densenet121"
