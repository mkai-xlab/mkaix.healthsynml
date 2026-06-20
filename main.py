import io
import os
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import numpy as np
import cv2
from contextlib import asynccontextmanager

from app.ml.model_registry import get_model
from app.ml.dataset import get_transforms
from app.ml.models.base_model import load_model_dict
from app.utils.s3_utils import s3_object_exists
from app.core.config import settings

# --- Model Loading Variables ---
MODEL = None
DEVICE = None
IMG_SIZE = 224
MODEL_NAME = "mobilenet_v2"  # Define model name globally

def load_best_model_from_s3(model: torch.nn.Module, model_name: str, device: torch.device):
    """
    Loads the best model weights from an S3 bucket.
    """
    model_sates_path = "models_states"
    model_save_dir = os.path.join(model_sates_path, model_name)
    best_model_key = os.path.join(model_save_dir, "best_model.pth").replace("\\", "/")

    print(f"Attempting to load best model for '{model_name}' from S3 key: '{best_model_key}'")

    if s3_object_exists(settings.AWS_S3_MODELS_BUCKET, best_model_key):
        try:
            # We only need to load the model state, so we pass optimizer=None
            load_model_dict(model=model, path=best_model_key, optimizer=None, device=device)
            print("Successfully loaded best model weights from S3.")
        except Exception as e:
            print(f"Error loading model weights from S3: {e}. The model will use its initial weights.")
    else:
        print(f"Warning: Best model checkpoint not found on S3 at '{best_model_key}'.")
        print("The model will use its initial (random or ImageNet pretrained) weights.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.
    Handles startup (loading model) and shutdown logic.
    """
    global MODEL, DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # --- Startup logic ---
        print("Starting up application and loading model...")
        # 1. Get the model architecture from the registry
        MODEL = get_model(MODEL_NAME, num_classes=5, pretrained=True) 
        
        # 2. Load our custom-trained best weights from S3
        load_best_model_from_s3(MODEL, MODEL_NAME, DEVICE)
        
        # 3. Move model to the correct device and set to evaluation mode
        MODEL.to(DEVICE)
        MODEL.eval()
        print(f"Model '{MODEL_NAME}' loaded successfully on device '{DEVICE}'.")
        
        yield  # The application runs while yielding
        
    except Exception as e:
        print(f"Fatal error during model startup: {e}")
        MODEL = None
        yield # Still yield so the app can start (though it might fail on requests)
    
    finally:
        # --- Shutdown logic ---
        print("Shutting down application. Cleaning up resources...")
        MODEL = None
        torch.cuda.empty_cache() # Clear GPU memory if applicable

# --- Application Setup ---
app = FastAPI(
    title="Knee Osteoarthritis KL-Grade Classification API",
    description="API for predicting Kellgren-Lawrence (KL) grade from knee X-ray images.",
    version="1.0.0",
    lifespan=lifespan # Register the lifespan context manager
)

# --- Image Pre-processing ---
_, val_transform = get_transforms(img_size=IMG_SIZE)

def preprocess_image(image_bytes: bytes):
    """
    Reads image bytes, applies transformations, and prepares it for the model.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image from bytes.")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = val_transform(img_rgb)
    return tensor.unsqueeze(0)

# --- API Endpoints ---
@app.get("/", tags=["General"])
def read_root():
    """Root endpoint to check if the API is running."""
    return {"message": "Welcome to the Knee OA Classification API!"}

@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Upload a knee X-ray image and get the raw prediction tensor.
    """
    if not MODEL:
        raise HTTPException(status_code=503, detail="Model is not loaded or failed to load.")

    image_bytes = await file.read()
    
    try:
        input_tensor = preprocess_image(image_bytes).to(DEVICE)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Image pre-processing failed: {e}")

    with torch.no_grad():
        logits = MODEL(input_tensor)
        
    confidence_scores = F.softmax(logits, dim=1)

    return {
        "filename": file.filename,
        "logits": logits.cpu().numpy().tolist()[0],
        "confidence_scores": confidence_scores.cpu().numpy().tolist()[0]
    }