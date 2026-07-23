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


class KneeOAPipeline:
    """Inference-only KL-grading pipeline for the native-CAM DenseNet checkpoint."""

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
        self.model_name = settings.DEFAULT_MODEL_NAME
        self.checkpoint_path = settings.MODEL_CHECKPOINT_PATH
        self.ordinal_type = settings.ORDINAL_TYPE
        self.checkpoint_metadata: dict = {}
        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        if self.model_name != "densenet121":
            raise RuntimeError(
                "This production pipeline requires DEFAULT_MODEL_NAME=densenet121"
            )
        if self.ordinal_type != "ce":
            raise RuntimeError("This checkpoint requires ORDINAL_TYPE=ce")

        checkpoint_path = os.path.abspath(self.checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Required model checkpoint was not mounted at {checkpoint_path}"
            )

        model = get_model(
            self.model_name,
            num_classes=5,
            pretrained=False,
            ordinal_type="ce",
        )
        print(f"Loading model checkpoint from {checkpoint_path}...")
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if not isinstance(checkpoint, dict):
            raise RuntimeError("Checkpoint must contain architecture metadata and weights")
        architecture = checkpoint.get("architecture")
        if architecture != settings.EXPECTED_MODEL_ARCHITECTURE:
            raise RuntimeError(
                f"Checkpoint architecture {architecture!r} does not match "
                f"{settings.EXPECTED_MODEL_ARCHITECTURE!r}"
            )
        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise RuntimeError("Checkpoint does not contain model_state_dict")
        model.load_state_dict(state_dict, strict=True)

        self.checkpoint_metadata = {
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
            "Model weights loaded successfully: "
            f"architecture={architecture}, epoch={checkpoint.get('epoch')}, "
            f"device={self.device}."
        )
        return model

    def postprocess(self, logits: torch.Tensor) -> dict:
        probabilities = F.softmax(logits.float(), dim=1)[0].cpu().numpy()
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
        logits, class_maps = inference_service.run_inference_with_class_maps(
            self.model, input_tensor
        )
        result = self.postprocess(logits)
        native_cam_image, _ = native_cam_service.generate_heatmap(
            model=self.model,
            class_maps=class_maps,
            processed_image=processed_image,
            predicted_class=result["predicted_class"],
        )

        # Preserve the established API contract while changing only the
        # implementation behind its historical heatmap field.
        result["gradcam_image"] = native_cam_image
        return result
