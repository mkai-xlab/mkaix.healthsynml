from fastapi import APIRouter

from app.services.prediction_service import prediction_service


router = APIRouter()


@router.get(
    "",
    summary="Check API health",
    description="Returns service readiness, active model mode, and device.",
)
def check_health():
    """Return a simple health check response with model and device information."""

    # Get the current model pipeline and return health information
    pipeline = prediction_service.pipeline
    return {
        "status": "healthy",
        "message": "Knee OA API is online",
        "model": pipeline.model_name,
        "device": str(pipeline.device),
    }
