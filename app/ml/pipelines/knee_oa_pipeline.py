import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision.transforms as transforms
from app.core.config import settings
from app.ml.model_registry import get_model
from app.ml.pipelines.gradcam import GradCAM
import base64

# --- Preprocessing classes ---

class SquarePadOpenCV(object):
    """Pads a rectangular X-ray image to a square, preserving the aspect ratio of the joint space."""
    def __call__(self, image: np.ndarray) -> np.ndarray:
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

class KneeOAPipeline:
    """
    Image preprocessing and inference pipeline for Knee Osteoarthritis Kellgren-Lawrence Grade prediction.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = settings.IMG_SIZE
        self.model_name = settings.DEFAULT_MODEL_NAME
        self.checkpoint_path = settings.MODEL_CHECKPOINT_PATH
        self.ordinal_type = settings.ORDINAL_TYPE
        
        # Define the validation transforms
        self.transform = transforms.Compose([
            SquarePadOpenCV(),
            OpenCVCLAHE(),
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load and prepare model
        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        """Initializes model from registry and loads weights from checkpoint."""
        # Get model from registry
        model = get_model(self.model_name, num_classes=5, pretrained=False, ordinal_type=self.ordinal_type)
        
        # Load checkpoint
        if os.path.exists(self.checkpoint_path):
            print(f"Loading model checkpoint from {self.checkpoint_path}...")
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    model.load_state_dict(checkpoint["model"])
                elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                print("Model weights loaded successfully.")
            except Exception as e:
                print(f"Error loading model weights from checkpoint: {e}. Using uninitialized weights.")
        else:
            print(f"Warning: Checkpoint not found at '{self.checkpoint_path}'. Inference will run with random weights.")
            
        model.to(self.device)
        model.eval()
        return model

    def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        """Converts raw image bytes to a preprocessed tensor ready for inference."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image from bytes. Ensure file is a valid image (PNG/JPEG).")
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img_rgb)
        return tensor.unsqueeze(0).to(self.device)

    def postprocess(self, logits: torch.Tensor) -> dict:
        """Converts model logits into predicted classes and detailed probability confidence scores."""
        # Apply sigmoid to get cumulative binary threshold probabilities
        probs_gt = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Ordinal class prediction: count how many binary thresholds (> 0.5) are met
        predicted_class = int(np.sum(probs_gt > 0.5))
        
        # Convert binary cumulative probabilities into individual class probabilities:
        # P(Class = 0) = 1 - P(Class > 0)
        # P(Class = k) = P(Class > k-1) - P(Class > k)
        # P(Class = 4) = P(Class > 3)
        p = np.zeros(5)
        p[0] = 1.0 - probs_gt[0]
        p[1] = probs_gt[0] - probs_gt[1]
        p[2] = probs_gt[1] - probs_gt[2]
        p[3] = probs_gt[2] - probs_gt[3]
        p[4] = probs_gt[3]
        
        # Clip negative differences (due to model noise) and normalize to ensure they sum to 1.0
        p = np.clip(p, 0.0, 1.0)
        p_sum = np.sum(p)
        if p_sum > 0:
            p = p / p_sum
        else:
            p = np.array([0.2, 0.2, 0.2, 0.2, 0.2]) # Fallback
            
        descriptions = {
            0: "Grade 0: Normal knee joint with no signs of osteoarthritis.",
            1: "Grade 1: Doubtful joint space narrowing and possible osteophytic lipping.",
            2: "Grade 2: Minimal/Definite osteophytes and possible joint space narrowing.",
            3: "Grade 3: Moderate multiple osteophytes, definite joint space narrowing, and some sclerosis.",
            4: "Grade 4: Severe large osteophytes, marked joint space narrowing, severe sclerosis, and definite deformity."
        }
        
        grade_labels = {
            0: "0Normal",
            1: "1Doubtful",
            2: "2Mild",
            3: "3Moderate",
            4: "4Severe"
        }
        
        confidence_details = {grade_labels[i]: float(p[i]) for i in range(5)}
        predicted_grade_label = grade_labels[predicted_class]
        
        return {
            "predicted_class": predicted_class,
            "predicted_grade": predicted_grade_label,
            "confidence": float(p[predicted_class]),
            "description": descriptions[predicted_class],
            "details": confidence_details
        }

    def generate_gradcam_base64(self, image_bytes: bytes, class_idx: int = None) -> str:
        """
        Generates a Grad-CAM heatmap overlayed on the preprocessed image and returns it as a Base64 string.
        """
        # Ensure gradient calculations are enabled for backward pass
        with torch.enable_grad():
            input_tensor = self.preprocess(image_bytes)
            # Enable requires_grad on input tensor to track backprop
            input_tensor.requires_grad = True
            
            # EfficientNet-B4 features layer 8 is the last convolutional block
            target_layer = self.model.model.features[8]
            
            gradcam = GradCAM(self.model, target_layer)
            try:
                cam, _ = gradcam.generate(input_tensor, class_idx)
            finally:
                gradcam.remove_hooks()
                
        # Re-read image bytes for visual processing
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image for Grad-CAM overlay.")
            
        # Standard preprocessing on original image (SquarePad + CLAHE)
        pad = SquarePadOpenCV()
        clahe = OpenCVCLAHE()
        img_processed = clahe(pad(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))
        img_resized = cv2.resize(img_processed, (self.img_size, self.img_size))
        
        # Apply Jet colormap to Grad-CAM heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap with alpha = 0.4
        alpha = 0.4
        overlay = cv2.addWeighted(img_resized, 1 - alpha, heatmap, alpha, 0)
        
        # Convert RGB back to BGR for OpenCV image encoding
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        
        # Encode overlay to JPEG bytes
        retval, buffer = cv2.imencode('.jpg', overlay_bgr)
        if not retval:
            raise ValueError("Could not encode overlay image to JPEG.")
            
        # Encode to Base64 string
        base64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_str}"

    def predict(self, image_bytes: bytes) -> dict:
        """Executes the complete preprocessing, inference, and Grad-CAM generation workflow."""
        input_tensor = self.preprocess(image_bytes)
        with torch.no_grad():
            logits = self.model(input_tensor)
        res = self.postprocess(logits)
        
        # Generate Grad-CAM for the predicted class
        predicted_class = res["predicted_class"]
        try:
            gradcam_base64 = self.generate_gradcam_base64(image_bytes, predicted_class)
            res["gradcam_image"] = gradcam_base64
        except Exception as e:
            print(f"Warning: Grad-CAM generation failed: {e}")
            res["gradcam_image"] = None
            
        return res

