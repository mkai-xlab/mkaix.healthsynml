"""
Tests for app.core.config.normalize_model_mode().

Input
-----
  - A string value submitted by the user or read from an env var
    (e.g. "dense_net_121", "seresnext50_32x4d")

Expected output
---------------
  - A canonical model-mode string used internally (e.g. "densenet121")
  - Or a ValueError if the string does not match any known alias
"""
import pytest

from app.core.config import normalize_model_mode


# ---------------------------------------------------------------------------
# Table-driven test: each (value, expected) pair describes one alias mapping.
# ---------------------------------------------------------------------------
# value      — what the user or env var might contain
# expected   — the canonical string that normalize_model_mode() must return
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # Exact canonical strings pass through unchanged
        ("densenet121", "densenet121"),
        # snake_case variant maps to the same canonical key
        ("dense_net_121", "densenet121"),
        # se_resnext aliases
        ("se_resnext",    "se_resnext"),
        ("seresnext50_32x4d", "se_resnext"),
        # ensemble is always the canonical key
        ("ensemble", "ensemble"),
    ],
)
def test_normalize_model_mode(value, expected):
    """Each known alias must be normalised to its expected canonical form."""
    assert normalize_model_mode(value) == expected


# ---------------------------------------------------------------------------
# Boundary test: a completely unknown value must raise ValueError.
# ---------------------------------------------------------------------------
def test_normalize_model_mode_rejects_unknown_value():
    """
    Input  : an arbitrary string that has no registered alias
    Output : raises ValueError whose message mentions "Unsupported MODEL_MODE"
    """
    with pytest.raises(ValueError, match="Unsupported MODEL_MODE"):
        normalize_model_mode("automatic")
