"""Stable public response schemas for the inference API."""

from typing import Optional

from pydantic import BaseModel, Field


class SinglePredictionResult(BaseModel):
    """One classifier result for a detected knee."""

    predicted_class: int = Field(..., description="Predicted KL grade index (0 to 4)")
    predicted_grade: str = Field(..., description="Predicted KL label, for example 2Mild")
    confidence: float = Field(..., description="Probability assigned to the predicted grade")
    description: str = Field(..., description="Medical description of the predicted grade")
    details: dict[str, float] = Field(..., description="Probability for every KL grade")
    box: Optional[list[int]] = Field(None, description="YOLO box [xmin, ymin, xmax, ymax]")
    yolo_confidence: float = Field(..., description="YOLO detection confidence score")
    knee_side: str = Field(..., description="right, left, or unknown")
    roi_image: Optional[str] = Field(None, description="Base64-encoded cropped ROI image")
    gradcam_image: str = Field(..., description="Base64-encoded Grad-CAM overlay")


class KneeOAPredictionResponse(BaseModel):
    """Response from ``POST /api/v1/predict``."""

    filename: str = Field(..., description="Name of the processed file")
    predictions: list[SinglePredictionResult] = Field(..., description="Result for each detected knee")
    annotated_image: str = Field(..., description="Base64-encoded full annotated X-ray")


class SingleDetectionResult(BaseModel):
    """One raw YOLO detection returned by the ROI-only endpoint."""

    box: list[int] = Field(..., description="YOLO box [xmin, ymin, xmax, ymax]")
    x: int = Field(..., description="Bounding box top-left x coordinate")
    y: int = Field(..., description="Bounding box top-left y coordinate")
    w: int = Field(..., description="Bounding box width")
    h: int = Field(..., description="Bounding box height")
    class_name: str = Field(..., description="YOLO class name")
    confidence: float = Field(..., description="YOLO detection confidence score")
    knee_side: str = Field(..., description="right, left, or unknown")
    roi_image: str = Field(..., description="Base64-encoded square ROI image")


class KneeDetectionResponse(BaseModel):
    """Response from ``POST /api/v1/predict/detect-roi``."""

    filename: str = Field(..., description="Name of the processed file")
    detected_image: str = Field(..., description="Base64-encoded annotated source X-ray")
    detections: list[SingleDetectionResult] = Field(..., description="Detected knee ROIs")
