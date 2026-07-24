from fastapi import APIRouter

from app.core.config import settings
from app.services.prediction_service import prediction_service


router = APIRouter()


@router.get("")
def get_model_info():
    pipeline = prediction_service.pipeline
    return {
        "model": pipeline.model_name,
        "architecture": pipeline.checkpoint_metadata["architecture"],
        "checkpoint": pipeline.checkpoint_paths,
        "loss": "cross_entropy",
        "input": {
            "resize": [settings.IMG_SIZE, settings.IMG_SIZE],
            "center_crop": [settings.CROP_SIZE, settings.CROP_SIZE],
            "laterality_canonicalization": True,
        },
        "heatmap": {
            "method": "native_class_activation_map",
            "source": (
                f"{pipeline.heatmap_model_name} five-map head for the selected grade"
            ),
            "gradient_free": True,
        },
    }
