"""Registry of classifier constructors supported by the inference service."""

from typing import Type

from app.ml.models.base_model import BaseModel
from app.ml.models.densenet121_model import DenseNet121Model
from app.ml.models.se_resnext50_32x4d_model import SEResNeXt50Model


# Keys here are checkpoint metadata values, not user-facing MODEL_MODE aliases.
MODEL_REGISTRY: dict[str, Type[BaseModel]] = {
    "densenet121": DenseNet121Model,
    "seresnext50_32x4d": SEResNeXt50Model,
}


def get_model(model_name: str, **kwargs) -> BaseModel:
    """Construct a classifier by its checkpoint-compatible registry key."""
    model_class = MODEL_REGISTRY.get(model_name)
    if model_class is None:
        raise ValueError(
            f"Model '{model_name}' not found in registry. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    print(f"Initializing model: '{model_name}'")
    return model_class(**kwargs)
