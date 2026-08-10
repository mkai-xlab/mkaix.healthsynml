import base64

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAMService:
    """Generate, assess, and render predicted-class Grad-CAM heatmaps.

    The service deliberately owns no model state. It uses each model's declared
    ``gradcam_target_layer`` so the visualization remains aligned with the
    checkpoint architecture selected by the pipeline.
    """

    @staticmethod
    def extract_gradcam( model: torch.nn.Module, input_tensor: torch.Tensor, predicted_class: int, output_size: tuple[int, int]) -> np.ndarray:
        """Generate Grad-CAM from the model's declared final feature layer.
        
        Args:
            model: The model to use for Grad-CAM.
            input_tensor: The input tensor to the model (image). 
            predicted_class: The class index for which to compute Grad-CAM.
            output_size: The size of the output heatmap (default (224, 224)).
        
        Returns:
            A 2D numpy array representing the Grad-CAM heatmap, normalized to [0, 1].

        """

        # get the target layer for Grad-CAM from the model
        target_layer = getattr(model, "gradcam_target_layer", None)
        if target_layer is None:
            raise TypeError("The configured model does not expose a Grad-CAM layer")


        # Capture the activation of the target layer during the forward pass 
        captured: dict[str, torch.Tensor] = {}


        def capture_activation(_module, _inputs, output):
            """Capture the activation of the target layer.
            
            Args:
                _module: The layer module being hooked (not used).
                _inputs: The inputs to the layer (not used).
                output: The output of the layer, which is the activation we want to capture.

            """

            captured["activation"] = output

        # The hook captures the final spatial features needed for Grad-CAM.
        handle = target_layer.register_forward_hook(capture_activation)
        try:

            # Set model to eval mode (turn off dropout, batchnorm, etc.) and zero gradients
            model.eval()

            # Clean up gradients
            model.zero_grad(set_to_none=True)

            # Clone the input tensor and ensure it requires gradients for backpropagation
            grad_input = input_tensor.detach().clone().requires_grad_(True)


            with torch.enable_grad():
                logits = model(grad_input)
                activation = captured.get("activation")


                if activation is None or not activation.requires_grad:
                    raise RuntimeError("Grad-CAM activation was not captured")

                # Compute the gradient of the predicted class score with respect to the activation
                gradient = torch.autograd.grad(
                    logits[0, int(predicted_class)], # Get the logit corresponding to the predicted class
                    activation, # activation maps from the target layer
                    retain_graph=False, # We don't need to retain the graph after this backward pass
                    create_graph=False, # We don't need to create a new graph for higher-order derivatives
                )[0]

                # Compute the weights for the activation maps by averaging the gradients across the spatial dimensions
                # [batch_size, channels, height, width] -> [batch_size, channels, 1, 1]
                weights = gradient.mean(dim=(2, 3), keepdim=True)

                # Compute the Grad-CAM by performing a weighted sum of the activation maps
                # [batch_size, channels, height, width] -> [batch_size, 1, height, width]
                cam = F.relu((weights * activation).sum(dim=1, keepdim=True))

                # Resize the CAM to the desired output size using bilinear interpolation
                cam = F.interpolate(
                    cam,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
        finally:

            # Remove the hook to avoid memory leaks and ensure that the model's state is not affected by this operation
            handle.remove()

        # Normalize the CAM to the range [0, 1] for visualization
        cam = cam.detach().float().cpu().numpy()
        maximum = float(cam.max())
        return cam / maximum if maximum > 1e-8 else np.zeros_like(cam)

    @staticmethod
    def render_heatmap(cam: np.ndarray, processed_image: np.ndarray) -> str:
        """Overlay an already selected map without changing its geometry."""

        # Convert the CAM to a heatmap using OpenCV's color mapping
        # COLORMAP_JET is a common choice for visualizing heatmaps, where low values are blue and high values are red.
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay the heatmap on the original image with specified weights for blending
        # Formula: output = alpha * image1 + beta * image2 + gamma (alpha and beta are weights, gamma is a scalar added to each sum)
        overlay = cv2.addWeighted(processed_image, 0.60, heatmap, 0.40, 0)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        
        success, buffer = cv2.imencode(".jpg", overlay_bgr)
        if not success:
            raise RuntimeError("Could not encode CAM overlay")
        encoded = base64.b64encode(buffer).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

gradcam_service = GradCAMService()
