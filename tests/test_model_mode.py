import pytest

from app.core.config import normalize_model_mode


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("densenet121", "densenet121"),
        ("dense_net_121", "densenet121"),
        ("se_resnext", "se_resnext"),
        ("seresnext50_32x4d", "se_resnext"),
        ("efficientnet", "efficientnet_b0"),
        ("efficientnet_b0", "efficientnet_b0"),
        ("ensemble", "ensemble"),
    ],
)
def test_normalize_model_mode(value, expected):
    assert normalize_model_mode(value) == expected


def test_normalize_model_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unsupported MODEL_MODE"):
        normalize_model_mode("automatic")
