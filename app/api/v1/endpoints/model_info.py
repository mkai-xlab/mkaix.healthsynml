from fastapi import APIRouter

from app.core.config import settings
from app.services.prediction_service import prediction_service


router = APIRouter()


@router.get("")
def get_model_info():
    pipeline = prediction_service.pipeline
    if pipeline.model_mode == "ensemble":
        heatmap = {
            "method": "dynamic_per_case_gradcam",
            "source": "selected component final convolutional feature layer",
            "gradient_free": False,
        }
    else:
        heatmap = {
            "method": "predicted_class_gradcam",
            "source": (
                f"{pipeline.heatmap_model_name} final convolutional feature layer"
            ),
            "gradient_free": False,
        }

    return {
        "model": pipeline.model_name,
        "architecture": pipeline.checkpoint_metadata["architecture"],
        "checkpoint": pipeline.checkpoint_paths,
        "loss": "cross_entropy",
        "input": {
            "resize": [settings.IMG_SIZE, settings.IMG_SIZE],
            "center_crop": None,
            "laterality_canonicalization": False,
        },
        "heatmap": heatmap,
    }
