from fastapi import APIRouter
from app.api.v1.endpoints import health, prediction, model_info

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["Health"])
api_router.include_router(prediction.router, prefix="/predict", tags=["Prediction"])
api_router.include_router(model_info.router, prefix="/models", tags=["Model Info"])
