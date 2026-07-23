import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.config import settings
from app.ml.model_registry import get_model
from app.services.gradcam_service import native_cam_service
from app.services.inference_service import inference_service
from app.services.preprocessing_service import preprocessing_service


def equal_soft_vote(logits: list[torch.Tensor]) -> torch.Tensor:
    """Average equally weighted class probabilities from compatible CE models."""
    if len(logits) != 2:
        raise ValueError("The configured ensemble requires exactly two logits tensors")
    if logits[0].shape != logits[1].shape:
        raise ValueError("Ensemble logits must have identical shapes")
    probabilities = [F.softmax(value.float(), dim=1) for value in logits]
    return torch.stack(probabilities, dim=0).mean(dim=0)


class KneeOAPipeline:
    """Inference-only equal-soft-voting KL-grading ensemble."""

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
        self.model_name = "densenet121+seresnext50_32x4d"
        self.ordinal_type = settings.ORDINAL_TYPE
        if self.ordinal_type != "ce":
            raise RuntimeError("Both ensemble checkpoints require ORDINAL_TYPE=ce")

        self.densenet_model, densenet_metadata = self._load_component(
            model_name="densenet121",
            checkpoint_path=settings.MODEL_CHECKPOINT_PATH,
            expected_architecture=settings.EXPECTED_MODEL_ARCHITECTURE,
        )
        self.se_resnext_model, se_resnext_metadata = self._load_component(
            model_name="seresnext50_32x4d",
            checkpoint_path=settings.SE_RESNEXT_CHECKPOINT_PATH,
            expected_architecture=settings.EXPECTED_SE_RESNEXT_ARCHITECTURE,
        )
        self.checkpoint_metadata = {
            "architecture": "equal_soft_voting_native_cam_ensemble",
            "epoch": {
                "densenet121": densenet_metadata["epoch"],
                "seresnext50_32x4d": se_resnext_metadata["epoch"],
            },
            "loss_type": "ce",
            "validation_metrics": {
                "densenet121": densenet_metadata["validation_metrics"],
                "seresnext50_32x4d": se_resnext_metadata["validation_metrics"],
            },
        }

    def _load_component(
        self,
        model_name: str,
        checkpoint_path: str,
        expected_architecture: str,
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
        if architecture != expected_architecture:
            raise RuntimeError(
                f"{model_name} checkpoint architecture {architecture!r} does not match "
                f"{expected_architecture!r}"
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
            "loss_type": checkpoint.get("loss_type"),
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
        densenet_logits, _ = inference_service.run_inference_with_class_maps(
            self.densenet_model, input_tensor
        )
        se_resnext_logits, se_resnext_maps = (
            inference_service.run_inference_with_class_maps(
                self.se_resnext_model, input_tensor
            )
        )
        probabilities = equal_soft_vote([densenet_logits, se_resnext_logits])
        result = self.postprocess(probabilities)
        native_cam_image, _ = native_cam_service.generate_heatmap(
            model=self.se_resnext_model,
            class_maps=se_resnext_maps,
            processed_image=processed_image,
            predicted_class=result["predicted_class"],
        )

        # Keep the established field while using the better-localized SE-ResNeXt CAM.
        result["gradcam_image"] = native_cam_image
        return result
