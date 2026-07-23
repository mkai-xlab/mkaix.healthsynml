import base64

import cv2
import numpy as np
import torch


class NativeCAMService:
    """Render the class map that is directly averaged into the predicted logit."""

    @staticmethod
    def _energy_metrics(cam: np.ndarray) -> dict[str, float]:
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
        total = float(cam.sum()) + 1e-8
        joint_energy = float(cam[joint].sum() / total)
        border_energy = float(cam[border].sum() / total)
        return {
            "joint_energy": joint_energy,
            "joint_enrichment": joint_energy / float(joint.mean()),
            "border_energy": border_energy,
            "border_enrichment": border_energy / float(border.mean()),
        }

    def generate_heatmap(
        self,
        model: torch.nn.Module,
        class_maps: torch.Tensor,
        processed_image: np.ndarray,
        predicted_class: int,
    ) -> tuple[str, dict[str, float]]:
        if not hasattr(model, "native_cam_from_class_maps"):
            raise TypeError("The configured model does not support native CAM")

        height, width = processed_image.shape[:2]
        cam_tensor = model.native_cam_from_class_maps(
            class_maps,
            class_index=predicted_class,
            output_size=(height, width),
        )
        cam = cam_tensor.detach().cpu().numpy()
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(processed_image, 0.60, heatmap, 0.40, 0)

        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode(".jpg", overlay_bgr)
        if not success:
            raise RuntimeError("Could not encode native-CAM overlay")
        encoded = base64.b64encode(buffer).decode("ascii")
        metrics = self._energy_metrics(cam)
        metrics["source_height"] = float(class_maps.shape[-2])
        metrics["source_width"] = float(class_maps.shape[-1])
        return f"data:image/jpeg;base64,{encoded}", metrics


native_cam_service = NativeCAMService()

# Compatibility import for existing clients/modules. This now generates native CAM.
gradcam_service = native_cam_service
