from typing import Type

from app.ml.models.base_model import BaseModel
from app.ml.models.densenet121_model import DenseNet121Model
from app.ml.models.se_resnext50_32x4d_model import SEResNeXt50Model
from app.ml.models.efficientnet_b0_model import EfficientNetB0Model

# Classifiers supported by the inference API and its local checkpoint layout.
MODEL_REGISTRY: dict[str, Type[BaseModel]] = {
    "densenet121": DenseNet121Model,
    "seresnext50_32x4d": SEResNeXt50Model,
    "efficientnet_b0": EfficientNetB0Model,
}

def get_model(model_name: str, **kwargs) -> BaseModel:
    """
    Retrieves and initializes a model from the registry by its string identifier.

    This function isolates model construction from the inference pipeline.

    Args:
        model_name (str): The unique identifier for the model as defined in MODEL_REGISTRY.
        **kwargs: Arbitrary keyword arguments to be passed to the model's constructor
                  (e.g., num_classes=5, pretrained=True).

    Returns:
        BaseModel: An initialized instance of the requested model.

    Raises:
        ValueError: If the provided model_name is not found in the registry.
    """
    model_class = MODEL_REGISTRY.get(model_name)
    
    if not model_class:
        raise ValueError(
            f"Model '{model_name}' not found in registry. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )
        
    print(f"Initializing model: '{model_name}'")
    return model_class(**kwargs)
