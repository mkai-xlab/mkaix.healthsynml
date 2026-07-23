from fastapi import APIRouter

from app.core.config import settings


router = APIRouter()


@router.get("")
def get_model_info():
    return {
        "model": settings.DEFAULT_MODEL_NAME,
        "architecture": settings.EXPECTED_MODEL_ARCHITECTURE,
        "checkpoint": settings.MODEL_CHECKPOINT_PATH,
        "loss": "cross_entropy",
        "input": {
            "resize": [settings.IMG_SIZE, settings.IMG_SIZE],
            "center_crop": [settings.CROP_SIZE, settings.CROP_SIZE],
            "laterality_canonicalization": True,
        },
        "heatmap": {
            "method": "native_class_activation_map",
            "source": "five 1x1-convolution grade maps",
            "gradient_free": True,
        },
    }
