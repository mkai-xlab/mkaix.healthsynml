import torch
import torch.nn.functional as F

from app.ml.pipelines.knee_oa_pipeline import equal_soft_vote


def test_equal_soft_vote_averages_probabilities_not_logits():
    densenet_logits = torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0]])
    se_resnext_logits = torch.tensor([[-1.0, 0.0, 3.0, 0.5, -0.5]])

    actual = equal_soft_vote([densenet_logits, se_resnext_logits])
    expected = (
        F.softmax(densenet_logits, dim=1)
        + F.softmax(se_resnext_logits, dim=1)
    ) / 2.0

    assert torch.allclose(actual, expected)
    assert torch.allclose(actual.sum(dim=1), torch.ones(1))
    assert not torch.allclose(
        actual, F.softmax((densenet_logits + se_resnext_logits) / 2.0, dim=1)
    )


def test_equal_soft_vote_rejects_incompatible_inputs():
    try:
        equal_soft_vote([torch.zeros(1, 5)])
    except ValueError as error:
        assert "exactly two" in str(error)
    else:
        raise AssertionError("Expected one-model voting to be rejected")
