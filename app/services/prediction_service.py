from app.ml.pipelines.knee_oa_pipeline import KneeOAPipeline

class PredictionService:
    """
    Service to handle business logic for Knee Osteoarthritis predictions.
    Orchestrates the ML prediction pipeline and structures results.
    """
    def __init__(self):
        # The pipeline handles model registration, loading, and inference.
        self.pipeline = KneeOAPipeline()

    def predict_image(self, file_name: str, image_bytes: bytes) -> dict:
        """
        Receives raw image bytes, detects ROIs with YOLOv8, runs them through 
        the classification pipeline, and returns a list of predictions alongside
        an annotated original image.
        """
        from app.services.roi_service import roi_service
        import cv2
        import numpy as np
        import base64
        
        # 1. Detect knees and get crops and coordinates
        knees = roi_service.detect_knees_with_coords(image_bytes)
        
        # 2. Decode original image to draw on
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        ai_results = []
        
        if not knees:
            # Fallback if no knees detected (or YOLO disabled)
            res = self.pipeline.predict(image_bytes)
            res["box"] = None
            res["yolo_confidence"] = 0.0
            res["knee_side"] = "unknown"
            res["roi_image"] = None
            ai_results.append(res)
        else:
            # Determine knee side (anatomical right/left) if exactly 2 ROIs are detected
            if len(knees) == 2:
                # Sort left-to-right by x coordinate (box[0])
                knees = sorted(knees, key=lambda k: k["box"][0])
                sides = ["right", "left"]
            else:
                sides = ["unknown"] * len(knees)
                
            for idx, knee in enumerate(knees):
                # Run prediction on the cropped knee image
                res = self.pipeline.predict(knee["crop_bytes"])
                res["box"] = knee["box"]
                res["yolo_confidence"] = knee["yolo_conf"]
                side = sides[idx]
                res["knee_side"] = side
                
                # Base64 encode the cropped ROI image
                roi_base64 = base64.b64encode(knee["crop_bytes"]).decode("utf-8")
                res["roi_image"] = f"data:image/png;base64,{roi_base64}"
                
                ai_results.append(res)
                
                # Draw bounding box and KL grade + side + confidence on original image
                x1, y1, x2, y2 = knee["box"]
                kl_grade = res.get("predicted_grade", "Unknown")
                conf = res.get("confidence", 0.0)
                
                # Bounding box color (Green)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                side_str = f" {side.upper()}" if side != "unknown" else ""
                label = f"{side_str} Grade {kl_grade} ({conf*100:.1f}%)".strip()
                
                # Draw text background for readability
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, max(15, y1 - 25)), (x1 + w, max(15, y1 - 25) + h + 5), (0, 255, 0), -1)
                cv2.putText(img, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
        # Encode annotated image back to base64
        _, buffer = cv2.imencode(".jpg", img)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        annotated_image_url = f"data:image/jpeg;base64,{img_base64}"
        
        return {
            "filename": file_name,
            "predictions": ai_results,
            "annotated_image": annotated_image_url
        }

# Singleton instance for the service layer
prediction_service = PredictionService()
