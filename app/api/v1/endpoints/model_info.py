from fastapi import APIRouter

from app.core.config import settings
from app.services.prediction_service import prediction_service


router = APIRouter()


@router.get(
    "",
    summary="Describe the active model",
    description="Returns the active checkpoint, preprocessing contract, and runtime Grad-CAM method.",
)
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

    # Some legacy checkpoints retain ``final_native_cam_ce`` in their metadata.
    # That is a training/checkpoint identifier; runtime explanations are Grad-CAM.
    runtime_architecture = (
        "two_model_weighted_soft_voting_gradcam_ensemble"
        if pipeline.model_mode == "ensemble"
        else f"{pipeline.model_name}_gradcam_ce"
    )
    return {
        "model": pipeline.model_name,
        "architecture": runtime_architecture,
        "checkpoint_architecture": pipeline.checkpoint_metadata["architecture"],
        "checkpoint": pipeline.checkpoint_paths,
        "loss": "cross_entropy",
        "input": {
            "resize": [settings.IMG_SIZE, settings.IMG_SIZE],
            "center_crop": None,
            "laterality_canonicalization": False,
        },
        "heatmap": heatmap,
    }
