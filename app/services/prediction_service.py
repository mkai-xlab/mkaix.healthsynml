"""Application service that joins YOLO detection and classifier predictions."""

import base64

import cv2
import numpy as np

from app.ml.pipelines.knee_oa_pipeline import KneeOAPipeline
from app.services.roi_service import (
    NO_KNEE_ROI_MESSAGE,
    assign_knee_sides,
    roi_service,
)


def _decode_source_image(image_bytes: bytes) -> np.ndarray:
    """Decode the original X-ray used for the response annotation."""
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image. Upload a valid PNG or JPEG image.")
    return image


def _encode_jpeg_data_url(image: np.ndarray) -> str:
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("Could not encode annotated image")
    encoded = base64.b64encode(buffer).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _draw_prediction_label(image: np.ndarray, knee: dict, result: dict) -> None:
    """Render the stable green annotation returned in the existing API contract."""
    x1, y1, x2, y2 = knee["box"]
    side = result["knee_side"]
    side_prefix = f"{side.upper()} " if side != "unknown" else ""
    label = (
        f"{side_prefix}Grade {result['predicted_grade']} "
        f"({result['confidence'] * 100:.1f}%)"
    )
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    text_y = max(15, y1 - 5)
    cv2.rectangle(
        image,
        (x1, text_y - 20),
        (x1 + text_width, text_y + text_height + 5),
        (0, 255, 0),
        -1,
    )
    cv2.putText(
        image,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
    )


class PredictionService:
    """Orchestrate ROI detection, KL inference, and response-image annotation."""

    def __init__(self) -> None:
        self.pipeline = KneeOAPipeline()

    def predict_image(self, file_name: str, image_bytes: bytes) -> dict:
        """Predict every detected knee and preserve the established JSON response."""
        source_image = _decode_source_image(image_bytes)
        knees = roi_service.detect_knees_with_coords(image_bytes)
        if not knees:
            # The concrete service raises this itself; retain a clear boundary guard.
            raise ValueError(NO_KNEE_ROI_MESSAGE)

        knees, sides = assign_knee_sides(knees, source_image.shape[1])
        predictions: list[dict] = []

        for knee, side in zip(knees, sides):
            prediction = self.pipeline.predict(knee["crop_bytes"], knee_side=side)
            prediction.update(
                {
                    "box": knee["box"],
                    "yolo_confidence": knee["yolo_conf"],
                    "knee_side": side,
                    "roi_image": (
                        "data:image/png;base64,"
                        f"{base64.b64encode(knee['crop_bytes']).decode('ascii')}"
                    ),
                }
            )
            predictions.append(prediction)
            _draw_prediction_label(source_image, knee, prediction)

        return {
            "filename": file_name,
            "predictions": predictions,
            "annotated_image": _encode_jpeg_data_url(source_image),
        }


# The API owns one pipeline instance so model weights are not reloaded per request.
prediction_service = PredictionService()
