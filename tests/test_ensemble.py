import torch
import torch.nn.functional as F

from app.ml.pipelines.knee_oa_pipeline import (
    select_heatmap_component,
    weighted_soft_vote,
)


def test_weighted_soft_vote_combines_probabilities_not_logits():
    densenet_logits = torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0]])
    se_resnext_logits = torch.tensor([[-1.0, 0.0, 3.0, 0.5, -0.5]])
    efficientnet_logits = torch.tensor([[0.0, 1.5, 0.5, -0.5, -1.0]])

    logits = {
        "densenet121": densenet_logits,
        "seresnext50_32x4d": se_resnext_logits,
        "efficientnet_b0": efficientnet_logits,
    }
    weights = {
        "densenet121": 0.50,
        "seresnext50_32x4d": 0.35,
        "efficientnet_b0": 0.15,
    }
    actual = weighted_soft_vote(logits, weights)
    expected = (
        F.softmax(densenet_logits, dim=1) * 0.50
        + F.softmax(se_resnext_logits, dim=1) * 0.35
        + F.softmax(efficientnet_logits, dim=1) * 0.15
    )

    assert torch.allclose(actual, expected)
    assert torch.allclose(actual.sum(dim=1), torch.ones(1))
    assert not torch.allclose(
        actual,
        F.softmax(
            (densenet_logits + se_resnext_logits + efficientnet_logits) / 3.0,
            dim=1,
        ),
    )


def test_weighted_soft_vote_rejects_incompatible_inputs():
    try:
        weighted_soft_vote({"only": torch.zeros(1, 5)}, {"only": 1.0})
    except ValueError as error:
        assert "at least two" in str(error)
    else:
        raise AssertionError("Expected one-model voting to be rejected")


def test_heatmap_source_uses_per_case_anatomy_not_model_average():
    probabilities = {
        "densenet121": torch.tensor([[0.05, 0.10, 0.55, 0.20, 0.10]]),
        "seresnext50_32x4d": torch.tensor([[0.05, 0.10, 0.50, 0.25, 0.10]]),
        "efficientnet_b0": torch.tensor([[0.05, 0.10, 0.20, 0.55, 0.10]]),
    }
    anatomy = {
        "densenet121": {
            "joint_energy": 0.65,
            "border_energy": 0.10,
            "lower_tibia_energy": 0.10,
            "peak_inside_joint": True,
            "anatomy_score": 0.5265,
        },
        "seresnext50_32x4d": {
            "joint_energy": 0.85,
            "border_energy": 0.05,
            "lower_tibia_energy": 0.05,
            "peak_inside_joint": True,
            "anatomy_score": 0.7671,
        },
        "efficientnet_b0": {
            "joint_energy": 0.80,
            "border_energy": 0.10,
            "lower_tibia_energy": 0.10,
            "peak_inside_joint": True,
            "anatomy_score": 0.6480,
        },
    }

    selected = select_heatmap_component(probabilities, 2, anatomy)

    assert selected == "seresnext50_32x4d"


def test_heatmap_source_does_not_force_bad_agreeing_map():
    probabilities = {
        "densenet121": torch.tensor([[0.47, 0.20, 0.15, 0.10, 0.08]]),
        "seresnext50_32x4d": torch.tensor([[0.39, 0.42, 0.10, 0.05, 0.04]]),
    }
    anatomy = {
        "densenet121": {
            "joint_energy": 0.37,
            "border_energy": 0.15,
            "lower_tibia_energy": 0.30,
            "peak_inside_joint": False,
            "anatomy_score": 0.2202,
        },
        "seresnext50_32x4d": {
            "joint_energy": 0.81,
            "border_energy": 0.16,
            "lower_tibia_energy": 0.08,
            "peak_inside_joint": True,
            "anatomy_score": 0.6260,
        },
    }

    assert select_heatmap_component(probabilities, 0, anatomy) == (
        "seresnext50_32x4d"
    )
