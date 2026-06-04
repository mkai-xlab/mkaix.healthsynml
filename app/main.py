from fastapi import FastAPI
from app.api.v1.router import api_router

app = FastAPI(
    title="MKAI Knee Osteoarthritis API",
    description="Ensemble model API for predicting Kellgren-Lawrence Grade from knee X-rays.",
    version="1.0.0"
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {
        "app": "MKAI Knee Osteoarthritis API",
        "docs_url": "/docs",
        "status": "healthy"
    }
