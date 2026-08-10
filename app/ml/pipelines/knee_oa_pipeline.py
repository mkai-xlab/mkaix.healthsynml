import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.config import settings
from app.ml.model_registry import get_model
from app.services.ensemble_service import ensemble_service
from app.services.gradcam_service import gradcam_service
from app.services.inference_service import inference_service
from app.services.preprocessing_service import preprocessing_service


class KneeOAPipeline:
    """Load the configured KL classifiers and generate predictions plus Grad-CAM.

    A single-model mode returns that model's softmax probabilities and Grad-CAM.
    Ensemble mode soft-votes probabilities, then uses the Grad-CAM from the model
    with the highest probability for the final predicted class.
    """

    component_config = {
        "densenet121": {
            "registry_name": "densenet121",
            "checkpoint_setting": "DENSENET121_CHECKPOINT_PATH",
        },
        "seresnext50_32x4d": {
            "registry_name": "seresnext50_32x4d",
            "checkpoint_setting": "SE_RESNEXT_CHECKPOINT_PATH",
        },
    }

    # weight of the component in the ensemble soft-voting; must sum to 1.0
    component_weights = {
        "densenet121": 0.55,
        "seresnext50_32x4d": 0.45,
    }

    # component names to load for each model mode; must be a subset of component_config keys
    mode_components = {
        "densenet121": ("densenet121",),
        "se_resnext": ("seresnext50_32x4d",),
        "ensemble": (
            "densenet121",
            "seresnext50_32x4d",
        ),
    }

    # Public response fields for each KL grade, used in the API output.
    grade_descriptions = {
        0: "Grade 0: Normal knee joint with no signs of osteoarthritis.",
        1: "Grade 1: Doubtful joint space narrowing and possible osteophytic lipping.",
        2: "Grade 2: Definite osteophytes and possible joint space narrowing.",
        3: "Grade 3: Multiple osteophytes, definite joint space narrowing, and some sclerosis.",
        4: "Grade 4: Large osteophytes, marked joint space narrowing, severe sclerosis, and deformity.",
    }
    grade_labels = {
        0: "0Normal",
        1: "1Doubtful",
        2: "2Mild",
        3: "3Moderate",
        4: "4Severe",
    }

    def __init__(self) -> None:

        # load device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load models and checkpoints based on the configured mode
        self.model_mode = settings.MODEL_MODE

        # load the models and their checkpoints
        component_names = self.mode_components[self.model_mode]
        self.ensemble_weights = {
            name: self.component_weights[name] for name in component_names
        }
        self.models = {}
        self.checkpoint_paths = {}

        # Load each component model and its checkpoint.
        for component_name in component_names:
            config = self.component_config[component_name]
            checkpoint_path = getattr(settings, config["checkpoint_setting"])
            model = self._load_component(
                model_name=config["registry_name"],
                checkpoint_path=checkpoint_path,
            )
            self.models[component_name] = model
            self.checkpoint_paths[component_name] = checkpoint_path

        self.model_name = "+".join(component_names)
        self.loss_function = "cross_entropy"
        if self.model_mode == "ensemble":
            self.heatmap_model_name = "highest_confidence_component"
        else:
            self.heatmap_model_name = component_names[0]


    def _load_component( self, model_name: str,checkpoint_path: str,) -> nn.Module:
        """Load a single model and its checkpoint, verifying architecture and loss type.

        Args:
            model_name: The name of the model architecture to load.
            checkpoint_path: The path to the model checkpoint file.
        Returns:
            The loaded PyTorch model with weights from the checkpoint.
        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            RuntimeError: If the checkpoint does not match the expected architecture or loss type.

        """

        # load the model architecture from the registry
        absolute_path = os.path.abspath(checkpoint_path)
        if not os.path.isfile(absolute_path):
            raise FileNotFoundError(
                f"Required {model_name} checkpoint was not mounted at {absolute_path}"
            )
        model = get_model(
            model_name,
            num_classes=5,
            pretrained=False,
        )
        print(f"Loading {model_name} checkpoint from {absolute_path}...")

        # load checkpoint
        try:
            checkpoint = torch.load(absolute_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(absolute_path, map_location=self.device)

        # if not a dict, return an error; otherwise, verify architecture and loss type
        if not isinstance(checkpoint, dict):
            raise RuntimeError("Checkpoint must contain architecture metadata and weights")

        # get the architecture and loss type from the checkpoint and verify them
        architecture = checkpoint.get("architecture")
        expected_architecture = model.architecture

        # verify that the architecture in the checkpoint matches the expected architecture
        if architecture != expected_architecture:
            raise RuntimeError(
                f"{model_name} checkpoint architecture {architecture!r} does not match "
                f"{expected_architecture!r}"
            )

        # verify that the loss type in the checkpoint is compatible with cross-entropy
        loss_type = checkpoint.get("loss_type")
        if loss_type not in {None, "ce", "cross_entropy"}:
            raise RuntimeError(
                f"{model_name} checkpoint loss_type={loss_type!r} is not CE-compatible"
            )

        # verify that the model name in the checkpoint matches the expected model name
        checkpoint_model_name = checkpoint.get("model_name")
        if checkpoint_model_name not in {None, model_name}:
            raise RuntimeError(
                f"Checkpoint declares model_name={checkpoint_model_name!r}, "
                f"expected {model_name!r}"
            )

        # load the model weights from the checkpoint
        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise RuntimeError("Checkpoint does not contain model_state_dict")

        # load model to device + weights, set to eval mode
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        print(f"{model_name} weights loaded successfully: device={self.device}.")
        return model


    def _postprocess_probabilities(self, probabilities: torch.Tensor) -> dict:
        """Convert one five-class probability vector into the public response fields.
        
        Args:
            probabilities: A tensor of shape (1, 5) containing the softmax probabilities for each KL grade.
        Returns:
            A dictionary containing the predicted class, predicted grade, confidence, description, and details. 

        """

        # probailities has 2 dimensions: (batch_size, 5). We want to convert it to a 1D numpy array of shape (5,).
        probabilities = probabilities[0].detach().cpu().numpy()

        # get the predicted class (index of the max probability) and the confidence details for each grade
        predicted_class = int(np.argmax(probabilities))
        confidence_details = {
            self.grade_labels[index]: float(probabilities[index])
            for index in range(5)
        }
        return {
            "predicted_class": predicted_class,
            "predicted_grade": self.grade_labels[predicted_class],
            "confidence": float(probabilities[predicted_class]),
            "description": self.grade_descriptions[predicted_class],
            "details": confidence_details,
        }

    def postprocess(self, probabilities: torch.Tensor) -> dict:
        """Backward-compatible name for callers outside the prediction path.

        Args:
            probabilities: A tensor of shape (1, 5) containing the softmax probabilities for each KL grade.
        Returns:
            A dictionary containing the predicted class, predicted grade, confidence, description, and details.

        """
        return self._postprocess_probabilities(probabilities)

    def _select_heatmap_component(self, component_probabilities: dict[str, torch.Tensor], predicted_class: int) -> str:
        """Use the single active model or the most confident ensemble map.
        
        Args:
            component_probabilities: A dictionary mapping component names to their softmax probability tensors.
            predicted_class: The index of the predicted class (0-4).
        Returns:
            The name of the component model to use for generating the Grad-CAM heatmap.

        """
        if len(self.models) == 1:
            return next(iter(self.models))
        return ensemble_service.select_heatmap_component(
            component_probabilities, predicted_class
        )

    def predict(self, image_bytes: bytes, knee_side: str = "unknown") -> dict:
        """Classify one square ROI and create a predicted-class Grad-CAM overlay.
        
        Args:
            image_bytes: The raw bytes of the input image.
            knee_side: Optional string indicating the side of the knee ("left", "right", or "unknown").
        Returns:
            A dictionary containing the predicted class, predicted grade, confidence, description, details, and Grad-CAM image.
        """

        # Build the model input: decode the knee ROI, apply CLAHE, square-pad, resize, normalize, and convert it to a tensor
        input_tensor, processed_image = preprocessing_service.preprocess_image(
            image_bytes
        )
        input_tensor = input_tensor.to(self.device)

        # Run inference for each model component and compute softmax probabilities
        logits = {
            name: inference_service.run_inference(model, input_tensor)
            for name, model in self.models.items()
        }

        # value has 2 dim: (batch_size, num_classes). We want to convert it to a 1D numpy array of shape (num_classes,) 
        component_probabilities = {
            name: F.softmax(value.float(), dim=1) for name, value in logits.items()
        }

        # If there's only one model, use its probabilities; otherwise, perform weighted soft voting for the ensemble
        probabilities = (
            next(iter(component_probabilities.values()))
            if len(logits) == 1
            else ensemble_service.weighted_soft_vote(logits, self.ensemble_weights)
        )



        result = self._postprocess_probabilities(probabilities)
        height, width = processed_image.shape[:2]

        # choice the component for Grad-CAM based on the predicted class and the model mode
        heatmap_component = self._select_heatmap_component(
            component_probabilities, result["predicted_class"]
        )

        # get heatmap
        heatmap = gradcam_service.extract_gradcam(
            model=self.models[heatmap_component],
            input_tensor=input_tensor,
            predicted_class=result["predicted_class"],
            output_size=(height, width),
        )

        # render the Grad-CAM heatmap on top of the processed image
        gradcam_image = gradcam_service.render_heatmap(
            heatmap, processed_image
        )

        # Keep the established response field for existing clients.
        result["gradcam_image"] = gradcam_image
        return result
