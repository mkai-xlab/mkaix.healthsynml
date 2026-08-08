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
    def extract_gradcam(
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        predicted_class: int,
        output_size: tuple[int, int],
    ) -> np.ndarray:
        """Generate Grad-CAM from the model's declared final feature layer."""
        target_layer = getattr(model, "gradcam_target_layer", None)
        if target_layer is None:
            raise TypeError("The configured model does not expose a Grad-CAM layer")

        captured: dict[str, torch.Tensor] = {}

        def capture_activation(_module, _inputs, output):
            captured["activation"] = output

        # The hook captures the final spatial features needed for Grad-CAM.
        handle = target_layer.register_forward_hook(capture_activation)
        try:
            model.eval()
            model.zero_grad(set_to_none=True)
            grad_input = input_tensor.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                logits = model(grad_input)
                activation = captured.get("activation")
                if activation is None or not activation.requires_grad:
                    raise RuntimeError("Grad-CAM activation was not captured")
                gradient = torch.autograd.grad(
                    logits[0, int(predicted_class)],
                    activation,
                    retain_graph=False,
                    create_graph=False,
                )[0]
                weights = gradient.mean(dim=(2, 3), keepdim=True)
                cam = F.relu((weights * activation).sum(dim=1, keepdim=True))
                cam = F.interpolate(
                    cam,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
        finally:
            handle.remove()

        cam = cam.detach().float().cpu().numpy()
        maximum = float(cam.max())
        return cam / maximum if maximum > 1e-8 else np.zeros_like(cam)

    @staticmethod
    def render_heatmap(cam: np.ndarray, processed_image: np.ndarray) -> str:
        """Overlay an already selected map without changing its geometry."""
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(processed_image, 0.60, heatmap, 0.40, 0)

        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode(".jpg", overlay_bgr)
        if not success:
            raise RuntimeError("Could not encode CAM overlay")
        encoded = base64.b64encode(buffer).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

gradcam_service = GradCAMService()
