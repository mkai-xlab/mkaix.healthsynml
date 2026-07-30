import math
import os
import base64
import numpy as np
import cv2
from app.core.config import settings


NO_KNEE_ROI_MESSAGE = (
    "No knee joint ROI was detected. Please upload a frontal knee X-ray "
    "with the complete tibiofemoral joint visible."
)
YOLO_ROI_EXPANSION = 1.15


def make_square_roi(
    image: np.ndarray, box: list[float] | tuple[float, float, float, float]
) -> np.ndarray:
    """Match the expanded square YOLO ROI geometry used for classifier training."""
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
    """Service handling Knee Joint ROI Detection and Cropping using YOLOv8."""
    
    def __init__(self, checkpoint_path: str = None):
        self.checkpoint_path = checkpoint_path or settings.YOLO_CHECKPOINT_PATH
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Lazy load YOLOv8 model weights."""
        abs_path = os.path.abspath(self.checkpoint_path)
        if not os.path.exists(abs_path):
            print(f"YOLOv8 checkpoint not found at: {abs_path}. ROI detection is disabled.")
            return
            
        try:
            from ultralytics import YOLO
            print(f"Loading YOLOv8 model from: {abs_path}...")
            self.model = YOLO(abs_path)
            print("YOLOv8 model loaded successfully.")
        except ImportError:
            print("ultralytics library is not installed. ROI detection is disabled. Install it via 'pip install ultralytics'.")
        except Exception as e:
            print(f"Failed to load YOLOv8 model: {e}")

    def detect_and_draw_boxes(self, image_bytes: bytes) -> tuple[str, list[dict]]:
        """Runs knee joint detection and returns base64 data URL of the image with drawn bounding boxes, and list of detections."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from bytes.")
        source_img = img.copy()

        detections = []
        if self.model is None:
            raise RuntimeError("YOLO ROI detector is unavailable.")
        else:
            # Run prediction
            results = self.model.predict(source=img, conf=0.45, save=False, verbose=False)
            boxes = results[0].boxes
            
            # Sort boxes by left-to-right (x1 coordinate)
            sorted_boxes = sorted(boxes, key=lambda b: float(b.xyxy[0][0]))
            
            if len(sorted_boxes) == 2:
                sides = ["right", "left"]
            else:
                sides = ["unknown"] * len(sorted_boxes)
            
            # Draw each box
            for idx, box in enumerate(sorted_boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names.get(class_id, "knee")
                side = sides[idx]
                
                # Draw box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Draw label
                label = f"{class_name} ({side}): {conf:.2f}" if side != "unknown" else f"{class_name}: {conf:.2f}"
                cv2.putText(img, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Match the training ROI while keeping the drawn display image separate.
                crop = make_square_roi(source_img, [x1, y1, x2, y2])
                _, crop_buffer = cv2.imencode(".png", crop)
                crop_base64 = base64.b64encode(crop_buffer).decode("utf-8")
                
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "x": x1,
                    "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "class_name": class_name,
                    "confidence": conf,
                    "knee_side": side,
                    "roi_image": f"data:image/png;base64,{crop_base64}"
                })

            if not detections:
                raise ValueError(NO_KNEE_ROI_MESSAGE)
                
        # Encode to base64
        _, buffer = cv2.imencode(".jpg", img)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{img_base64}", detections

    def detect_knees_with_coords(self, image_bytes: bytes) -> list[dict]:
        """Runs knee joint detection and returns list of dicts with box coords and crop bytes."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from bytes.")
            
        if self.model is None:
            raise RuntimeError("YOLO ROI detector is unavailable.")
            
        results = self.model.predict(source=img, conf=0.45, save=False, verbose=False)
        boxes = results[0].boxes
        
        sorted_boxes = sorted(boxes, key=lambda b: float(b.xyxy[0][0]))
        
        knees = []
        for box in sorted_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            
            # Match the expanded square ROI used during paired-view training.
            crop = make_square_roi(img, [x1, y1, x2, y2])
            
            # Encode back to bytes
            _, buffer = cv2.imencode(".png", crop)
            
            knees.append({
                "box": [x1, y1, x2, y2],
                "crop_bytes": buffer.tobytes(),
                "yolo_conf": conf
            })

        if not knees:
            raise ValueError(NO_KNEE_ROI_MESSAGE)

        return knees

# Global instance of ROI Service
roi_service = ROIService()
