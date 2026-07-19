import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from app.services.preprocessing_service import SquarePadOpenCV, OpenCVCLAHE

class GradCAM:
    """
    Computes Grad-CAM class activation heatmaps for a given convolutional target layer.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None
        
        # Register hooks
        self.hook_forward = self.target_layer.register_forward_hook(self._save_features)
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.hook_backward = self.target_layer.register_full_backward_hook(self._save_gradients)
        else:
            self.hook_backward = self.target_layer.register_backward_hook(self._save_gradients)
            
    def _save_features(self, module, input, output):
        self.features = output
        
    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x: torch.Tensor, class_idx: int = None, ordinal_type: str = "focal_corn") -> tuple[np.ndarray, int]:
        self.model.eval()
        self.model.zero_grad()
        
        # Run forward pass enabling gradients temporarily
        with torch.enable_grad():
            output = self.model(x)
            
            # Determine class_idx if not provided
            if class_idx is None:
                sigmoids = torch.sigmoid(output).cpu().detach().numpy()[0]
                if ordinal_type in ["corn", "focal_corn"]:
                    p = np.zeros(5)
                    p[0] = 1.0 - sigmoids[0]
                    cumprod = 1.0
                    for i in range(1, 4):
                        cumprod *= sigmoids[i-1]
                        p[i] = cumprod * (1.0 - sigmoids[i])
                    p[4] = cumprod * sigmoids[3]
                    class_idx = int(np.argmax(p))
                elif ordinal_type == "threshold":
                    class_idx = int(np.sum(sigmoids > 0.5))
                else:
                    class_idx = int(torch.argmax(output, dim=1).item())
                    
            self.model.zero_grad()
            
            # Determine loss index depending on loss type (ordinal vs categorical)
            if ordinal_type == "ce":
                loss = output[0, class_idx]
            else:
                task_idx = min(max(class_idx - 1, 0), output.size(1) - 1)
                loss = output[0, task_idx]
                
            loss.backward()
            
        # Pool the gradients across channels
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Apply weights to features
        cam = torch.sum(weights * self.features, dim=1).squeeze(0)
        
        # Retain only positive features via ReLU
        cam = F.relu(cam)
        cam = cam.cpu().detach().numpy()
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
            
        return cam, class_idx

    def remove_hooks(self):
        """Clean up hooks to prevent memory leaks."""
        self.hook_forward.remove()
        self.hook_backward.remove()

class GradCAMService:
    """Service to create Grad-CAM heatmap overlays on medical knee images."""
    def generate_heatmap(self, model: nn.Module, input_tensor: torch.Tensor, img_rgb: np.ndarray, predicted_class: int, ordinal_type: str) -> str:
        """
        Generates visual activation map, overlays it on the image and returns base64 string.
        """
        # Determine the last feature map/convolution layer to hook based on model class name
        model_class_name = model.__class__.__name__
        target_layer = None
        
        # Find norm5 layer for DenseNet or conv_head for EfficientNet/MobileNet
        if "DenseNet" in model_class_name:
            if hasattr(model, "model") and hasattr(model.model, "features"):
                target_layer = getattr(model.model.features, "norm5", None)
            elif hasattr(model, "backbone"):
                # Find norm5 inside timm features or fallback to last BatchNorm2d in the backbone
                target_layer = getattr(model.backbone, "norm5", None)
                if target_layer is None:
                    for module in reversed(list(model.backbone.modules())):
                        if isinstance(module, nn.BatchNorm2d):
                            target_layer = module
                            break
        elif "EfficientNet" in model_class_name:
            if hasattr(model, "model"):
                target_layer = getattr(model.model, "conv_head", None)
        elif "MobileNet" in model_class_name:
            if hasattr(model, "model"):
                target_layer = getattr(model.model, "conv_head", None)
            
        # Fallback: search for last convolution or batchnorm module
        if target_layer is None:
            for module in reversed(list(model.modules())):
                if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                    target_layer = module
                    break
                    
        if target_layer is None:
            return ""
            
        # Run Grad-CAM
        gradcam = GradCAM(model, target_layer)
        try:
            cam, _ = gradcam(input_tensor, class_idx=predicted_class, ordinal_type=ordinal_type)
        except Exception as e:
            print(f"Error computing Grad-CAM: {e}")
            return ""
        finally:
            gradcam.remove_hooks()
            
        # Prep original image for layout (SquarePad & CLAHE) to match clinical viewing
        pad = SquarePadOpenCV()
        clahe = OpenCVCLAHE()
        img_processed = clahe(pad(img_rgb))
        
        # Get processed high-resolution dimensions
        h_p, w_p = img_processed.shape[:2]
        
        # Resize Grad-CAM mask to high resolution (matching original/padded image size)
        cam_resized = cv2.resize(cam, (w_p, h_p))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend original high-resolution processed image and heatmap
        alpha = 0.4
        overlay = cv2.addWeighted(img_processed, 1.0 - alpha, heatmap, alpha, 0)
        
        # Encode overlay to base64
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', overlay_bgr)
        base64_str = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/jpeg;base64,{base64_str}"

# Singleton instance
gradcam_service = GradCAMService()
