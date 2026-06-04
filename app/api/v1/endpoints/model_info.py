from fastapi import APIRouter

router = APIRouter()

@router.get("")
def get_model_info():
    return {
        "ensemble_weights": {
            "efficientnet_b0": 0.4,
            "densenet121": 0.4,
            "mobilenet_v2": 0.2
        },
        "available_models": ["efficientnet_b0", "densenet121", "mobilenet_v2"]
    }
