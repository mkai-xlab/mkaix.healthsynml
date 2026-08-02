from fastapi import APIRouter

from app.services.prediction_service import prediction_service


router = APIRouter()


@router.get(
    "",
    summary="Check API health",
    description="Returns service readiness, active model mode, device, and checkpoint metadata.",
)
def check_health():
    pipeline = prediction_service.pipeline
    return {
        "status": "healthy",
        "message": "Knee OA API is online",
        "model": pipeline.model_name,
        "device": str(pipeline.device),
        "checkpoint": pipeline.checkpoint_metadata,
    }
