import io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import cv2
from contextlib import asynccontextmanager
import torchvision.transforms as transforms
import uvicorn

try:
    import timm
except ImportError:
    raise ImportError("The 'timm' library is required to run this API. Install it using: pip install timm")

# --- Configuration & Paths ---
# You can set the environment variable MODEL_PATH to point to your specific checkpoint.
MODEL_PATH = os.environ.get("MODEL_PATH", "checkpoints/se_resnext50_checkpoints/best_model.pth")
IMG_SIZE = 310

MODEL = None
DEVICE = None

# --- Model Definition ---
class SEResNeXtModel(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = False):
        super(SEResNeXtModel, self).__init__()
        self.model = timm.create_model('seresnext50_32x4d', pretrained=pretrained, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# --- Image Preprocessing ---
class SquarePadOpenCV(object):
    """Pads a rectangular X-ray image to a square, preserving the aspect ratio of the joint space."""
    def __call__(self, image):
        h, w = image.shape[:2]
        max_wh = max(h, w)
        pad_top = (max_wh - h) // 2
        pad_bottom = max_wh - h - pad_top
        pad_left = (max_wh - w) // 2
        pad_right = max_wh - w - pad_left
        
        padded_image = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right, 
            borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        return padded_image

class OpenCVCLAHE(object):
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance bone textures."""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(img_lab)
        clahe_l_channel = clahe.apply(l_channel)
        merged_lab_image = cv2.merge((clahe_l_channel, a_channel, b_channel))
        return cv2.cvtColor(merged_lab_image, cv2.COLOR_LAB2RGB)

# Inference Preprocessing Pipeline
val_transform = transforms.Compose([
    SquarePadOpenCV(),
    OpenCVCLAHE(),
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Decodes image bytes, applies CLAHE and Square Padding, and returns normalized tensor."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image from bytes. Ensure the file is a valid image (PNG, JPG, JPEG).")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = val_transform(img_rgb)
    return tensor.unsqueeze(0)  # Add batch dimension [1, 3, 310, 310]

# --- Lifecycle Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles API startup and shutdown logic, loading model weights into memory."""
    global MODEL, DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 1. Initialize model structure
    MODEL = SEResNeXtModel(num_classes=5, pretrained=False)

    # 2. Load trained checkpoints
    if os.path.exists(MODEL_PATH):
        print(f"Loading checkpoint from '{MODEL_PATH}'...")
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                MODEL.load_state_dict(checkpoint["model"])
                print(f"Successfully loaded model weights from checkpoint (Epoch {checkpoint.get('epoch', 'N/A')}).")
            elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                MODEL.load_state_dict(checkpoint["model_state_dict"])
                print(f"Successfully loaded model weights from checkpoint (Epoch {checkpoint.get('epoch', 'N/A')}).")
            else:
                MODEL.load_state_dict(checkpoint)
                print("Successfully loaded model weights directly from state_dict.")
        except Exception as e:
            print(f"Error loading checkpoint weights: {e}. Starting with default initialization.")
    else:
        print(f"Warning: Checkpoint file not found at '{MODEL_PATH}'. Model will use initialized weights.")

    MODEL.to(DEVICE)
    MODEL.eval()
    
    yield  # API is running
    
    # Clean up
    MODEL = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Application shutdown complete.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="SE-ResNeXt-50 Knee Osteoarthritis Classification API",
    description="FastAPI service for predicting Kellgren-Lawrence (KL) grade (0-4) from knee X-ray images using SE-ResNeXt-50.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Label Meanings ---
KL_GRADE_DESCRIPTIONS = {
    0: "Grade 0: Normal knee joint with no signs of osteoarthritis.",
    1: "Grade 1: Doubtful joint space narrowing and possible osteophytic lipping.",
    2: "Grade 2: Minimal/Definite osteophytes and possible joint space narrowing.",
    3: "Grade 3: Moderate multiple osteophytes, definite joint space narrowing, and some sclerosis.",
    4: "Grade 4: Severe large osteophytes, marked joint space narrowing, severe sclerosis, and definite deformity."
}

# --- API Routes ---
@app.get("/")
def read_root():
    """Simple healthcheck endpoint."""
    return {
        "status": "online",
        "model": "se_resnext50_32x4d",
        "device": str(DEVICE) if DEVICE else "unknown",
        "checkpoint_loaded": MODEL_PATH
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an uploaded knee X-ray image and returns the predicted KL grade and confidence scores.
    """
    if not MODEL:
        raise HTTPException(status_code=503, detail="Model is not loaded or failed to initialize.")

    image_bytes = await file.read()
    
    try:
        input_tensor = preprocess_image(image_bytes).to(DEVICE)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {e}")

    with torch.no_grad():
        logits = MODEL(input_tensor)
        probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
        
    predicted_class = int(np.argmax(probabilities))
    
    # Formulate confidence scores dict mapping class to confidence percentage
    confidence_scores = {str(i): float(probabilities[i]) for i in range(5)}
    
    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "predicted_class_name": KL_GRADE_DESCRIPTIONS[predicted_class],
        "confidence_scores": confidence_scores
    }

# --- Python Client Caller Example ---
# If you want to call this API from another Python script, use the following snippet:
#
# ```python
# import requests
#
# url = "http://127.0.0.1:8000/predict"
# files = {"file": open("path_to_your_knee_xray.png", "rb")}
#
# response = requests.post(url, files=files)
# print(response.json())
# ```

if __name__ == "__main__":
    uvicorn.run("api_inference:app", host="127.0.0.1", port=8000, reload=True)
