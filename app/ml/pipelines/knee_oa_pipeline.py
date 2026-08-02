import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.config import settings
from app.ml.model_registry import get_model
from app.services.ensemble_service import (
    MAX_HEATMAP_BORDER_ENERGY,
    MAX_HEATMAP_LOWER_TIBIA_ENERGY,
    MIN_HEATMAP_JOINT_ENERGY,
    ensemble_service,
)
from app.services.gradcam_service import gradcam_service
from app.services.inference_service import inference_service
from app.services.preprocessing_service import preprocessing_service


logger = logging.getLogger(__name__)

class KneeOAPipeline:
    """Environment-selected single-model or soft-voting KL pipeline."""

    component_config = {
        "densenet121": {
            "registry_name": "densenet121",
            "checkpoint_setting": "MODEL_CHECKPOINT_PATH",
        },
        "seresnext50_32x4d": {
            "registry_name": "seresnext50_32x4d",
            "checkpoint_setting": "SE_RESNEXT_CHECKPOINT_PATH",
        },
        "efficientnet_b0": {
            "registry_name": "efficientnet_b0",
            "checkpoint_setting": "EFFICIENTNET_B0_CHECKPOINT_PATH",
        },
    }
    component_weights = {
        "densenet121": 0.55,
        "seresnext50_32x4d": 0.45,
        "efficientnet_b0": 0.0,
    }
    mode_components = {
        "densenet121": ("densenet121",),
        "se_resnext": ("seresnext50_32x4d",),
        "efficientnet_b0": ("efficientnet_b0",),
        "ensemble": (
            "densenet121",
            "seresnext50_32x4d",
        ),
    }

    descriptions = {
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

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_mode = settings.MODEL_MODE

        component_names = self.mode_components[self.model_mode]
        self.ensemble_weights = {
            name: self.component_weights[name] for name in component_names
        }
        self.models = {}
        self.checkpoint_paths = {}
        metadata = {}
        for component_name in component_names:
            config = self.component_config[component_name]
            checkpoint_path = getattr(settings, config["checkpoint_setting"])
            model, component_metadata = self._load_component(
                model_name=config["registry_name"],
                checkpoint_path=checkpoint_path,
            )
            self.models[component_name] = model
            self.checkpoint_paths[component_name] = checkpoint_path
            metadata[component_name] = component_metadata

        self.model_name = "+".join(component_names)
        if self.model_mode == "ensemble":
            self.heatmap_model_name = "dynamic_per_case_anatomy_gate"
            self.checkpoint_metadata = {
                "architecture": "two_model_weighted_soft_voting_gradcam_ensemble",
                "epoch": {
                    name: value["epoch"] for name, value in metadata.items()
                },
                "loss_type": "ce",
                "validation_metrics": {
                    name: value["validation_metrics"]
                    for name, value in metadata.items()
                },
                "weights": self.ensemble_weights,
            }
        else:
            self.heatmap_model_name = component_names[0]
            self.checkpoint_metadata = metadata[component_names[0]]

    def _load_component(
        self,
        model_name: str,
        checkpoint_path: str,
    ) -> tuple[nn.Module, dict]:
        absolute_path = os.path.abspath(checkpoint_path)
        if not os.path.isfile(absolute_path):
            raise FileNotFoundError(
                f"Required {model_name} checkpoint was not mounted at {absolute_path}"
            )

        model = get_model(
            model_name,
            num_classes=5,
            pretrained=False,
            ordinal_type="ce",
        )
        print(f"Loading {model_name} checkpoint from {absolute_path}...")
        try:
            checkpoint = torch.load(
                absolute_path, map_location=self.device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(absolute_path, map_location=self.device)

        if not isinstance(checkpoint, dict):
            raise RuntimeError("Checkpoint must contain architecture metadata and weights")
        architecture = checkpoint.get("architecture")
        expected_architecture = model.architecture
        if architecture != expected_architecture:
            raise RuntimeError(
                f"{model_name} checkpoint architecture {architecture!r} does not match "
                f"{expected_architecture!r}"
            )
        loss_type = checkpoint.get("loss_type")
        if loss_type not in {None, "ce", "cross_entropy"}:
            raise RuntimeError(
                f"{model_name} checkpoint loss_type={loss_type!r} is not CE-compatible"
            )
        checkpoint_model_name = checkpoint.get("model_name")
        if checkpoint_model_name not in {None, model_name}:
            raise RuntimeError(
                f"Checkpoint declares model_name={checkpoint_model_name!r}, "
                f"expected {model_name!r}"
            )
        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise RuntimeError("Checkpoint does not contain model_state_dict")
        model.load_state_dict(state_dict, strict=True)

        metadata = {
            "architecture": architecture,
            "epoch": checkpoint.get("epoch"),
            "loss_type": loss_type or "ce",
            "validation_metrics": {
                key: value
                for key, value in checkpoint.get("validation_metrics", {}).items()
                if key not in {"probas", "report"}
            },
        }
        model.to(self.device)
        model.eval()
        print(
            f"{model_name} weights loaded successfully: "
            f"architecture={architecture}, epoch={metadata['epoch']}, "
            f"device={self.device}."
        )
        return model, metadata

    def postprocess(self, probabilities: torch.Tensor) -> dict:
        probabilities = probabilities[0].detach().cpu().numpy()
        predicted_class = int(np.argmax(probabilities))
        confidence_details = {
            self.grade_labels[index]: float(probabilities[index])
            for index in range(5)
        }
        return {
            "predicted_class": predicted_class,
            "predicted_grade": self.grade_labels[predicted_class],
            "confidence": float(probabilities[predicted_class]),
            "description": self.descriptions[predicted_class],
            "details": confidence_details,
        }

    def predict(self, image_bytes: bytes, knee_side: str = "unknown") -> dict:
        input_tensor, processed_image, was_mirrored = (
            preprocessing_service.preprocess_image(image_bytes, knee_side=knee_side)
        )
        input_tensor = input_tensor.to(self.device)
        logits = {
            name: inference_service.run_inference(model, input_tensor)
            for name, model in self.models.items()
        }
        component_probabilities = {
            name: F.softmax(value.float(), dim=1) for name, value in logits.items()
        }
        probabilities = (
            next(iter(component_probabilities.values()))
            if len(logits) == 1
            else ensemble_service.weighted_soft_vote(logits, self.ensemble_weights)
        )
        result = self.postprocess(probabilities)
        height, width = processed_image.shape[:2]
        component_cams = {
            name: gradcam_service.extract_gradcam(
                model=model,
                input_tensor=input_tensor,
                predicted_class=result["predicted_class"],
                output_size=(height, width),
            )
            for name, model in self.models.items()
        }
        anatomy_metrics = {
            name: gradcam_service.energy_metrics(cam)
            for name, cam in component_cams.items()
        }
        heatmap_component = (
            next(iter(self.models))
            if len(self.models) == 1
            else ensemble_service.select_heatmap_component(
                component_probabilities,
                result["predicted_class"],
                anatomy_metrics,
            )
        )
        selected_metrics = anatomy_metrics[heatmap_component]
        if not (
            selected_metrics["joint_energy"] >= MIN_HEATMAP_JOINT_ENERGY
            and selected_metrics["border_energy"] <= MAX_HEATMAP_BORDER_ENERGY
            and selected_metrics["lower_tibia_energy"]
            <= MAX_HEATMAP_LOWER_TIBIA_ENERGY
            and selected_metrics["peak_inside_joint"]
        ):
            logger.warning(
                "No CAM candidate passed the anatomy gate; using best "
                "available component=%s metrics=%s",
                heatmap_component,
                selected_metrics,
            )
        gradcam_image = gradcam_service.render_heatmap(
            component_cams[heatmap_component], processed_image
        )

        # Keep the established response field for existing clients.
        result["gradcam_image"] = gradcam_image
        return result
