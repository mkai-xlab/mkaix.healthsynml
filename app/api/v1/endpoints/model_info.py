from fastapi import APIRouter

from app.core.config import settings


router = APIRouter()


@router.get("")
def get_model_info():
    return {
        "model": "densenet121+seresnext50_32x4d",
        "architecture": "equal_soft_voting_native_cam_ensemble",
        "checkpoint": {
            "densenet121": settings.MODEL_CHECKPOINT_PATH,
            "seresnext50_32x4d": settings.SE_RESNEXT_CHECKPOINT_PATH,
        },
        "loss": "cross_entropy",
        "input": {
            "resize": [settings.IMG_SIZE, settings.IMG_SIZE],
            "center_crop": [settings.CROP_SIZE, settings.CROP_SIZE],
            "laterality_canonicalization": True,
        },
        "heatmap": {
            "method": "native_class_activation_map",
            "source": "SE-ResNeXt five-map head for the ensemble-selected grade",
            "gradient_free": True,
        },
    }
