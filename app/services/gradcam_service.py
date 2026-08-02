import base64

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAMService:
    """Generate, assess, and render post-hoc Grad-CAM heatmaps."""

    @staticmethod
    def energy_metrics(cam: np.ndarray) -> dict[str, float]:
        """Measure whether one case's positive evidence is anatomically plausible."""
        height, width = cam.shape
        joint = np.zeros_like(cam, dtype=bool)
        joint[
            int(0.28 * height) : int(0.72 * height),
            int(0.06 * width) : int(0.94 * width),
        ] = True
        border = np.ones_like(cam, dtype=bool)
        border[
            int(0.08 * height) : int(0.92 * height),
            int(0.08 * width) : int(0.92 * width),
        ] = False
        lower_tibia = np.zeros_like(cam, dtype=bool)
        lower_tibia[
            int(0.72 * height) : int(0.96 * height),
            int(0.06 * width) : int(0.94 * width),
        ] = True
        total = float(cam.sum()) + 1e-8
        joint_energy = float(cam[joint].sum() / total)
        border_energy = float(cam[border].sum() / total)
        lower_tibia_energy = float(cam[lower_tibia].sum() / total)
        peak_y, peak_x = np.unravel_index(int(np.argmax(cam)), cam.shape)
        peak_inside_joint = bool(joint[peak_y, peak_x] and total > 1e-7)
        anatomy_score = (
            joint_energy
            * (1.0 - border_energy)
            * (1.0 - lower_tibia_energy)
        )
        return {
            "joint_energy": joint_energy,
            "joint_enrichment": joint_energy / float(joint.mean()),
            "border_energy": border_energy,
            "border_enrichment": border_energy / float(border.mean()),
            "lower_tibia_energy": lower_tibia_energy,
            "peak_x": float(peak_x / max(width - 1, 1)),
            "peak_y": float(peak_y / max(height - 1, 1)),
            "peak_inside_joint": peak_inside_joint,
            "anatomy_score": anatomy_score,
        }

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
