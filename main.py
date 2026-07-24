import os
import torch
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.
    Handles startup and shutdown logic.
    """
    print("Starting up Knee Osteoarthritis API service...")
    yield
    print("Shutting down Knee Osteoarthritis API service...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- Application Setup ---
app = FastAPI(
    title="Knee Osteoarthritis KL-Grade Classification API",
    description="API for predicting Kellgren-Lawrence (KL) grade from knee X-ray images.",
    version="2.0.0",
    lifespan=lifespan
)

# Register API v1 routes
app.include_router(api_router, prefix="/api/v1")

# Configure CORS to allow cross-origin requests from any client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---
@app.get("/", tags=["General"])
def read_root():
    """Root endpoint to check if the API is running."""
    return {"message": "Welcome to the Knee OA Classification API!"}
