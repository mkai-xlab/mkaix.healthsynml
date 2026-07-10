from typing import Type
from app.ml.models.base_model import BaseModel
from app.ml.models.efficientnet_b0_model import EfficientNetB0Model
from app.ml.models.efficientnet_b4_model import EfficientNetB4Model
from app.ml.models.densenet121_model import DenseNet121Model
from app.ml.models.mobilenet_v2_model import MobileNetV2Model
from app.ml.models.densenet201_model import DenseNet201Model

# MODEL_REGISTRY acts as a central directory for all available models in the application.
# It maps a string identifier to the corresponding model class.
MODEL_REGISTRY: dict[str, Type[BaseModel]] = {
    "efficientnet_b0": EfficientNetB0Model,
    "efficientnet_b4": EfficientNetB4Model,
    "densenet121": DenseNet121Model,
    "mobilenet_v2": MobileNetV2Model,
    "densenet201": DenseNet201Model,
}

def get_model(model_name: str, **kwargs) -> BaseModel:
    """
    Retrieves and initializes a model from the registry by its string identifier.

    This function provides a flexible way to instantiate different models
    without needing to import their classes directly in the main training script.

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
