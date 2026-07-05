import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) generator
    designed to hook a convolutional network's target layer and output a normalized heatmap.
    Supports both standard classification and multi-task threshold classification (Frank-Hall/CORAL).
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None
        
        # Register forward and backward hooks
        self.hook_forward = self.target_layer.register_forward_hook(self.save_features)
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.hook_backward = self.target_layer.register_full_backward_hook(self.save_gradients)
        else:
            self.hook_backward = self.target_layer.register_backward_hook(self.save_gradients)
            
    def save_features(self, module, input, output):
        self.features = output
        
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate(self, input_tensor: torch.Tensor, class_idx: int = None) -> tuple[np.ndarray, int]:
        """
        Generates the Grad-CAM heatmap for a given input tensor and target class.
        """
        # Run forward pass (enable gradient computation for backprop)
        output = self.model(input_tensor)
        
        if class_idx is None:
            if output.shape[1] == 4:
                # Ordinal threshold logits: count predictions exceeding 0.5
                probs = torch.sigmoid(output)
                class_idx = int(torch.sum(probs > 0.5).item())
            else:
                class_idx = torch.argmax(output, dim=1).item()
                
        # Reset model gradients
        self.model.zero_grad()
        
        # Compute loss/score for the target class
        if output.shape[1] == 4:
            # For Frank-Hall ordinal classification, we backpropagate from the sub-task logit
            # corresponding to the predicted class threshold (clamped to available tasks).
            task_idx = min(max(class_idx - 1, 0), 3)
            loss = output[0, task_idx]
        else:
            loss = output[0, class_idx]
            
        # Run backward pass
        loss.backward()
        
        # Pool gradients globally across height and width
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Compute weighted sum of features
        cam = torch.sum(weights * self.features, dim=1).squeeze(0)
        
        # Apply ReLU to keep only features with positive influence on prediction
        cam = F.relu(cam)
        cam = cam.cpu().detach().numpy()
        
        # Normalize the heatmap between 0.0 and 1.0
        if cam.max() > 0:
            cam = cam / cam.max()
            
        # Resize to input tensor width and height
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        
        return cam, class_idx
        
    def remove_hooks(self):
        """Removes forward and backward hooks to prevent memory leaks."""
        self.hook_forward.remove()
        self.hook_backward.remove()
