from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class SinglePredictionResult(BaseModel):
    predicted_class: int = Field(..., description="The predicted KL grade class index (0 to 4)")
    predicted_grade: str = Field(..., description="The label of predicted KL grade (e.g. 0Normal, 2Mild)")
    confidence: float = Field(..., description="The model's confidence for the predicted class")
    description: str = Field(..., description="Medical description of the predicted grade")
    details: Dict[str, float] = Field(..., description="Detailed class probabilities")
    box: Optional[List[int]] = Field(None, description="YOLO bounding box [xmin, ymin, xmax, ymax]")
    yolo_confidence: float = Field(..., description="YOLO detection confidence score")
    knee_side: str = Field(..., description="Anatomical side of the knee (right, left, or unknown)")
    roi_image: Optional[str] = Field(None, description="Base64 encoded cropped ROI image")
    gradcam_image: str = Field(..., description="Base64 encoded Grad-CAM activation heatmap image")

class KneeOAPredictionResponse(BaseModel):
    filename: str = Field(..., description="Name of the processed file")
    predictions: List[SinglePredictionResult] = Field(..., description="List of predictions for each detected knee")
    annotated_image: str = Field(..., description="Base64 encoded full annotated X-ray image")

class SingleDetectionResult(BaseModel):
    box: List[int] = Field(..., description="YOLO bounding box [xmin, ymin, xmax, ymax]")
    x: int = Field(..., description="Bounding box top-left x coordinate")
    y: int = Field(..., description="Bounding box top-left y coordinate")
    w: int = Field(..., description="Bounding box width")
    h: int = Field(..., description="Bounding box height")
    class_name: str = Field(..., description="YOLO class name")
    confidence: float = Field(..., description="YOLO detection confidence score")
    knee_side: str = Field(..., description="Anatomical side of the knee (right, left, or unknown)")
    roi_image: str = Field(..., description="Base64 encoded cropped ROI image")

class KneeDetectionResponse(BaseModel):
    filename: str = Field(..., description="Name of the processed file")
    detected_image: str = Field(..., description="Base64 encoded full annotated X-ray image")
    detections: List[SingleDetectionResult] = Field(..., description="List of detected ROIs")
