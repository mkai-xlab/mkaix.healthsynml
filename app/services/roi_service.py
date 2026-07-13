import os
import base64
import numpy as np
import cv2
from app.core.config import settings

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

    def detect_and_draw_boxes(self, image_bytes: bytes) -> str:
        """Runs knee joint detection and returns base64 data URL of the image with drawn bounding boxes."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from bytes.")
            
        if self.model is None:
            # Fallback: Draw a dummy box in the center if model is not loaded for testing
            h, w = img.shape[:2]
            cv2.rectangle(img, (int(w*0.1), int(h*0.1)), (int(w*0.9), int(h*0.9)), (0, 0, 255), 3)
            cv2.putText(img, "YOLOv8 Not Loaded - Dummy Box", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # Run prediction
            results = self.model.predict(source=img, conf=0.45, save=False, verbose=False)
            boxes = results[0].boxes
            
            # Draw each box
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                
                # Draw box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Draw label
                label = f"Knee Joint: {conf:.2f}"
                cv2.putText(img, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
        # Encode to base64
        _, buffer = cv2.imencode(".jpg", img)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{img_base64}"

    def crop_knees(self, image_bytes: bytes) -> list[bytes]:
        """Runs knee joint detection and returns list of cropped image bytes for downstream classification."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from bytes.")
            
        if self.model is None:
            # Fallback: return full image if model not available
            return [image_bytes]
            
        results = self.model.predict(source=img, conf=0.45, save=False, verbose=False)
        boxes = results[0].boxes
        
        # Sort boxes by left-to-right coordinate so we keep a consistent ordering (e.g. left knee first)
        sorted_boxes = sorted(boxes, key=lambda b: float(b.xyxy[0][0]))
        
        crops = []
        for box in sorted_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Crop image
            crop = img[y1:y2, x1:x2]
            
            # Encode back to bytes
            _, buffer = cv2.imencode(".png", crop)
            crops.append(buffer.tobytes())
            
        # If no knees detected, fallback to the original image
        if not crops:
            return [image_bytes]
            
        return crops

# Global instance of ROI Service
roi_service = ROIService()
