"""YOLOv8 detection and the square ROI contract shared with classifier training."""

import base64
import math
import os
import cv2
import numpy as np
from app.core.config import settings


NO_KNEE_ROI_MESSAGE = (
    "No knee joint ROI was detected. Please upload a frontal knee X-ray "
    "with the complete tibiofemoral joint visible." )
YOLO_ROI_EXPANSION = 1.15
YOLO_CONFIDENCE_THRESHOLD = 0.45


def assign_knee_sides(
    knees: list[dict], image_width: int
) -> tuple[list[dict], list[str]]:
    """Order detected knees and infer side labels for the API response."""

    # if there are 2 knees, sort them from left to right
    if len(knees) == 2:
        ordered_knees = sorted(knees, key=lambda knee: knee["box"][0])
        return ordered_knees, ["right", "left"]

    # if there are not 2 knees, return the knees with the unknown side
    if len(knees) != 1:
        return knees, ["unknown"] * len(knees)

    # if there is only one knee, determine the side based on the center x-coordinate
    center_x = (knees[0]["box"][0] + knees[0]["box"][2]) / 2
    if center_x < image_width * 0.40:
        return knees, ["right"]
    if center_x > image_width * 0.60:
        return knees, ["left"]
    return knees, ["unknown"]


def make_square_roi( image: np.ndarray, box: list[float] | tuple[float, float, float, float]) -> np.ndarray:
    """Expand a YOLO box to a square and pad only where it reaches an image edge.
    
    Args:
        image: The original image as a NumPy array.
        box: A YOLO bounding box in the format [x1, y1, x2, y2] where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

    Returns:
        A square crop of the image as a NumPy array, padded with black pixels if necessary

    """
    

    # Calculate the target x1,y1,x2,y2 point

    height, width = image.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    box_width, box_height = x2 - x1, y2 - y1
    if box_width <= 0 or box_height <= 0:
        raise ValueError(f"Invalid YOLO box: {box}")

    
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    side = int(math.ceil(max(box_width, box_height) * YOLO_ROI_EXPANSION))
    wanted_x1 = int(math.floor(center_x - side / 2))
    wanted_y1 = int(math.floor(center_y - side / 2))
    wanted_x2, wanted_y2 = wanted_x1 + side, wanted_y1 + side


    # crop the image [y1:y2, x1:x2] and pad with black if the box is outside the image
    crop = image[
        max(0, wanted_y1) : min(height, wanted_y2),
        max(0, wanted_x1) : min(width, wanted_x2),
    ]
    if crop.size == 0:
        raise ValueError(f"YOLO box does not overlap the image: {box}")

    return cv2.copyMakeBorder(
        crop,
        max(0, -wanted_y1),
        max(0, wanted_y2 - height),
        max(0, -wanted_x1),
        max(0, wanted_x2 - width),
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


class ROIService:
    """Load YOLO once and expose the same square crop to every API consumer."""

    def __init__(self, checkpoint_path: str | None = None) -> None:
        self.checkpoint_path = checkpoint_path or settings.YOLO_CHECKPOINT_PATH
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the configured detector; requests fail clearly if it is unavailable."""
        checkpoint_path = os.path.abspath(self.checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            print(f"YOLOv8 checkpoint not found at: {checkpoint_path}. ROI detection is disabled.")
            return

        try:
            from ultralytics import YOLO

            print(f"Loading YOLOv8 model from: {checkpoint_path}...")
            self.model = YOLO(checkpoint_path)
            print("YOLOv8 model loaded successfully.")
        except ImportError:
            print("ultralytics is not installed. ROI detection is disabled.")
        except Exception as error:
            print(f"Failed to load YOLOv8 model: {error}")

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        """ Decode bytes to a NumPy array suitable for OpenCV processing"""

        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image. Upload a valid PNG or JPEG image.")
        return image

    def _detect_sorted_boxes(self, image: np.ndarray) -> list:
        """Run a single YOLO pass and sort a bilateral study from image-left to image-right."""

        # if the model is not loaded, raise an error
        if self.model is None:
            raise RuntimeError("YOLO ROI detector is unavailable.")

        # run a single YOLO pass and sort a bilateral study from image-left to image-right
        results = self.model.predict(
            source=image,
            conf=YOLO_CONFIDENCE_THRESHOLD,
            save=False,
            verbose=False,
        )


        # results[0]: first input image, results[0].boxes: detected boxes for that image
        # box.xyxy -> [x1,y1,x2,y2]
        return sorted(results[0].boxes, key=lambda box: float(box.xyxy[0][0]))

    @staticmethod
    def _box_values(box) -> tuple[list[int], float, int]:

        # box.xyxy -> [x1,y1,x2,y2]
        coordinates = [int(value) for value in box.xyxy[0].tolist()]
        return coordinates, float(box.conf[0]), int(box.cls[0])

    @staticmethod
    def _encode_data_url(image: np.ndarray, extension: str, mime_type: str) -> str:
        """ convert to encode-64 format """

        success, buffer = cv2.imencode(extension, image)
        if not success:
            raise RuntimeError(f"Could not encode {mime_type} image")
        encoded = base64.b64encode(buffer).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def detect_and_draw_boxes(self, image_bytes: bytes) -> tuple[str, list[dict]]:
        """Return an annotated source image and display-ready square crops."""

        source_image = self._decode_image(image_bytes)
        annotated_image = source_image.copy()


        boxes = self._detect_sorted_boxes(source_image)
        box_data = [self._box_values(box) for box in boxes]
        _, sides = assign_knee_sides(
            [{"box": coordinates} for coordinates, _, _ in box_data],
            source_image.shape[1],
        )
        detections: list[dict] = []

        for (coordinates, confidence, class_id), side in zip(box_data, sides):
            x1, y1, x2, y2 = coordinates
            class_name = self.model.names.get(class_id, "knee")
            label = (
                f"{class_name} ({side}): {confidence:.2f}"
                if side != "unknown"
                else f"{class_name}: {confidence:.2f}"
            )
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                annotated_image,
                label,
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            crop = make_square_roi(source_image, coordinates)
            detections.append(
                {
                    "box": coordinates,
                    "x": x1,
                    "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "class_name": class_name,
                    "confidence": confidence,
                    "knee_side": side,
                    "roi_image": self._encode_data_url(crop, ".png", "image/png"),
                }
            )

        if not detections:
            raise ValueError(NO_KNEE_ROI_MESSAGE)

        return self._encode_data_url(annotated_image, ".jpg", "image/jpeg"), detections

    def detect_knees_with_coords(self, image_bytes: bytes) -> list[dict]:
        """Return classifier-ready PNG crops and their original YOLO box metadata."""

        # decode the image bytes to a numpy array
        source_image = self._decode_image(image_bytes)

        # detect the knees with their coordinates
        boxes = self._detect_sorted_boxes(source_image)

        # create a list to store the knees with their coordinates and the confidence
        knees: list[dict] = []

        for box in boxes:
            coordinates, confidence, _ = self._box_values(box)

            # make a square ROI of the knee
            crop = make_square_roi(source_image, coordinates)
            # encode the ROI to a PNG image
            success, buffer = cv2.imencode(".png", crop)
            if not success:
                raise RuntimeError("Could not encode knee ROI")
            knees.append(
                {
                    "box": coordinates,
                    "crop_bytes": buffer.tobytes(),
                    "yolo_conf": confidence,
                }
            )

        if not knees:
            raise ValueError(NO_KNEE_ROI_MESSAGE)
        return knees


# Shared process-wide detector. The checkpoint is loaded once when the API starts.
roi_service = ROIService()
