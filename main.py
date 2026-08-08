from pathlib import Path

import torch
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.api.v1.router import api_router


REPOSITORY_ROOT = Path(__file__).resolve().parent
TOOLS_HOME = REPOSITORY_ROOT / "tools" / "unified_dashboard" / "index.html"
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

app = FastAPI(
    title="Knee Osteoarthritis KL-Grade Classification API",
    description="API for predicting Kellgren-Lawrence (KL) grade from knee X-ray images.",
    version="2.0.0",
    lifespan=lifespan
)

app.include_router(api_router, prefix="/api/v1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---
@app.get(
    "/",
    tags=["General"],
    summary="Show API welcome message",
    description="Confirms that the Knee OA API process is reachable.",
)
def read_root():
    """Root endpoint to check if the API is running."""
    return {"message": "Welcome to the Knee OA Classification API!"}


@app.get("/tools", include_in_schema=False)
def tools_dashboard() -> FileResponse:
    """Single entry point for the prediction, response, and augmentation tools."""
    return FileResponse(TOOLS_HOME)


@app.get(
    "/result-viewer",
    tags=["Result Viewer"],
    summary="Open the prediction result viewer",
    description="Browser viewer for rendering a complete JSON response from POST /api/v1/predict.",
    include_in_schema=False,
)
def response_viewer_page() -> FileResponse:
    return FileResponse(RESPONSE_VIEWER_PAGE)


@app.get("/result-viewer/{asset_path:path}", include_in_schema=False)
def response_viewer_asset(asset_path: str) -> FileResponse:
    """Serve the existing viewer CSS/JS without changing its JSON behavior."""
    viewer_root = REPOSITORY_ROOT / "tools" / "kl_response_viewer"
    candidate = (viewer_root / asset_path).resolve()
    if viewer_root.resolve() not in candidate.parents or not candidate.is_file():
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Viewer asset not found")
    return FileResponse(candidate)


# Backward-compatible aliases for existing bookmarks. The viewer is now served
# by this API container; no second viewer container is required.
@app.get("/tools/viewer", include_in_schema=False)
def legacy_response_viewer_page() -> FileResponse:
    return response_viewer_page()


@app.get("/tools/viewer/{asset_path:path}", include_in_schema=False)
def legacy_response_viewer_asset(asset_path: str) -> FileResponse:
    return response_viewer_asset(asset_path)
