from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas.prediction import KneeDetectionResponse, KneeOAPredictionResponse
from app.services.prediction_service import prediction_service
from app.services.roi_service import roi_service


router = APIRouter()


@router.post(
    "",
    response_model=KneeOAPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict KL grades",
    description="Detects knee joints, predicts KL grades 0-4, and returns probabilities, ROI images, and predicted-class Grad-CAM heatmaps.",
)
async def predict_knee_oa(
    file: UploadFile = File(..., description="Knee X-ray PNG or JPEG file"),
):
    if not file:
        raise HTTPException(status_code=400, detail="No file was uploaded.")
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        result = prediction_service.predict_image(file.filename, image_bytes)
        return KneeOAPredictionResponse(**result)
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500, detail=f"Inference pipeline error: {error}"
        ) from error


@router.post(
    "/detect-roi",
    response_model=KneeDetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect knee ROIs",
    description="Runs YOLOv8 and returns detected knee boxes, square ROI images, and an annotated source image.",
)
async def detect_knee_roi(
    file: UploadFile = File(..., description="Knee X-ray PNG or JPEG file"),
):
    if not file:
        raise HTTPException(status_code=400, detail="No file was uploaded.")
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        detected_image_url, detections = roi_service.detect_and_draw_boxes(image_bytes)
        return KneeDetectionResponse(
            filename=file.filename,
            detected_image=detected_image_url,
            detections=detections,
        )
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500, detail=f"ROI detection error: {error}"
        ) from error
