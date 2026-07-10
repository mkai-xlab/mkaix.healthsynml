import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from app.core.config import settings
from app.ml.model_registry import get_model
from app.services.preprocessing_service import preprocessing_service
from app.services.inference_service import inference_service
from app.services.gradcam_service import gradcam_service

class KneeOAPipeline:
    """
    Image preprocessing and inference pipeline for Knee Osteoarthritis Kellgren-Lawrence Grade prediction.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = settings.DEFAULT_MODEL_NAME
        self.checkpoint_path = settings.MODEL_CHECKPOINT_PATH
        self.ordinal_type = settings.ORDINAL_TYPE
        
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
                try:
                    checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                except TypeError:
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

    def postprocess(self, logits: torch.Tensor) -> dict:
        """Converts model logits into predicted classes and detailed probability confidence scores."""
        # Apply sigmoid to get raw probabilities
        sigmoids = torch.sigmoid(logits).cpu().numpy()[0]
        
        p = np.zeros(5)
        
        # Check ordinal type (supports CORN/Focal CORN and CORAL/threshold)
        is_corn = self.ordinal_type in ["corn", "focal_corn"]
        
        if is_corn:
            # CORN chain rule formula:
            # P(y = 0) = 1 - p1
            # P(y = k) = p1 * p2 * ... * pk-1 * (1 - pk)
            # P(y = 4) = p1 * p2 * p3 * p4
            p[0] = 1.0 - sigmoids[0]
            cumprod = 1.0
            for i in range(1, 4):
                cumprod *= sigmoids[i - 1]
                p[i] = cumprod * (1.0 - sigmoids[i])
            p[4] = cumprod * sigmoids[3]
            
            # Predict class by Argmax over CORN probabilities
            predicted_class = int(np.argmax(p))
        else:
            # CORAL (Rank Ordinal / Threshold) cumulative difference formula:
            p[0] = 1.0 - sigmoids[0]
            p[1] = sigmoids[0] - sigmoids[1]
            p[2] = sigmoids[1] - sigmoids[2]
            p[3] = sigmoids[2] - sigmoids[3]
            p[4] = sigmoids[3]
            
            # Clip and normalize CORAL probabilities (since they can be negative due to model noise)
            p = np.clip(p, 0.0, 1.0)
            p_sum = np.sum(p)
            if p_sum > 0:
                p = p / p_sum
            else:
                p = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
                
            # Predict class by counting thresholds > 0.5
            predicted_class = int(np.sum(sigmoids > 0.5))
            
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
        """Executes the complete modular preprocessing, inference, and Grad-CAM workflow."""
        # 1. Preprocessing (runs OpenCV CLAHE, SquarePad and turns to PIL/Tensor)
        input_tensor, img_rgb = preprocessing_service.preprocess_image(image_bytes)
        input_tensor = input_tensor.to(self.device)
        
        # 2. Inference (runs model forward pass under no_grad)
        logits = inference_service.run_inference(self.model, input_tensor)
        
        # 3. Postprocessing (calculates class probabilities based on CORN/CORAL)
        result = self.postprocess(logits)
        
        # 4. Grad-CAM generation (creates overlay and encodes to base64)
        predicted_class = result["predicted_class"]
        gradcam_img = gradcam_service.generate_heatmap(
            model=self.model,
            input_tensor=input_tensor,
            img_rgb=img_rgb,
            predicted_class=predicted_class,
            ordinal_type=self.ordinal_type
        )
        
        result["gradcam_image"] = gradcam_img
        return result
