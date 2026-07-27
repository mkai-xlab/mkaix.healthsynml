import os
import secrets
from pathlib import Path

import torch
from fastapi import FastAPI, File, Query, UploadFile
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from scripts.roi_augmentation_preview.app import (
    decode_image,
    augment_roi,
)
from app.api.v1.router import api_router


REPOSITORY_ROOT = Path(__file__).resolve().parent
TOOLS_HOME = REPOSITORY_ROOT / "tools" / "unified_dashboard" / "index.html"
AUGMENTATION_PAGE = (
    REPOSITORY_ROOT / "scripts" / "roi_augmentation_preview" / "index.html"
)
RESPONSE_VIEWER_PAGE = REPOSITORY_ROOT / "tools" / "kl_response_viewer" / "index.html"

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


@app.get("/tools", include_in_schema=False)
def tools_dashboard() -> FileResponse:
    """Single entry point for the prediction, response, and augmentation tools."""
    return FileResponse(TOOLS_HOME)


@app.get("/tools/augmentation", include_in_schema=False)
def augmentation_page() -> FileResponse:
    return FileResponse(AUGMENTATION_PAGE)


@app.get("/tools/viewer", include_in_schema=False)
def response_viewer_page() -> FileResponse:
    return FileResponse(RESPONSE_VIEWER_PAGE)


@app.get("/tools/viewer/{asset_path:path}", include_in_schema=False)
def response_viewer_asset(asset_path: str) -> FileResponse:
    """Serve the existing viewer CSS/JS without changing its JSON behavior."""
    viewer_root = REPOSITORY_ROOT / "tools" / "kl_response_viewer"
    candidate = (viewer_root / asset_path).resolve()
    if viewer_root.resolve() not in candidate.parents or not candidate.is_file():
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Viewer asset not found")
    return FileResponse(candidate)


@app.post(
    "/augment",
    include_in_schema=False,
    responses={200: {"content": {"image/png": {}}}},
    response_class=Response,
)
async def augment_roi_image(
    image: UploadFile = File(...),
    seed: int | None = Query(default=None, ge=0, le=2**32 - 1),
) -> Response:
    """Return one visible PNG produced by the current training augmentation."""
    payload = await image.read(20 * 1024 * 1024 + 1)
    source_rgb = decode_image(payload)
    effective_seed = seed if seed is not None else secrets.randbits(32)
    augmented_rgb, operations = augment_roi(source_rgb, effective_seed)
    import cv2

    success, encoded = cv2.imencode(
        ".png", cv2.cvtColor(augmented_rgb, cv2.COLOR_RGB2BGR)
    )
    if not success:
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail="Could not encode augmented image")
    return Response(
        content=encoded.tobytes(),
        media_type="image/png",
        headers={
            "X-Augmentation-Seed": str(effective_seed),
            "X-Augmentation-Operations": "; ".join(operations),
            "Content-Disposition": f'inline; filename="augmented_{effective_seed}.png"',
        },
    )
