from fastapi import APIRouter

from app.services.prediction_service import prediction_service


router = APIRouter()


@router.get(
    "",
    summary="Describe the active model",
    description="Returns the active model and loss function.",
)
def get_model_info():
    """Return the active model and loss function."""

    pipeline = prediction_service.pipeline
    return {
        "model": pipeline.model_name,
        "loss": pipeline.loss_function,
    }
