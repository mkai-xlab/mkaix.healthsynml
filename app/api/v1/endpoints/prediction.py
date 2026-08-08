from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.api.dependencies import read_uploaded_image
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
) -> KneeOAPredictionResponse:
    """Detect one or two knees, then return a KL prediction for each ROI.
    Args:
        file: An uploaded knee X-ray image in PNG or JPEG format.
    Returns:
        A KneeOAPredictionResponse containing the predicted KL grades, probabilities, ROI images,
        and Grad-CAM heatmaps for each detected knee.
    Raises:
        HTTPException: If the uploaded file is invalid or if an error occurs during prediction.
    """


    try:
        filename, image_bytes = await read_uploaded_image(file)

        # call the prediction service to get the results
        result = prediction_service.predict_image(filename, image_bytes)

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
) -> KneeDetectionResponse:
    """Return the deterministic YOLO ROI crops used by the classifier.
    Args:
        file: An uploaded knee X-ray image in PNG or JPEG format.
    Returns:
        A KneeDetectionResponse containing the detected knee boxes, square ROI images, and an annotated source
        image with bounding boxes drawn.
    Raises:
        HTTPException: If the uploaded file is invalid or if an error occurs during ROI detection.
    """
    try:
        # read the uploaded image and validate it
        filename, image_bytes = await read_uploaded_image(file)

        # call the ROI service to detect knees and draw bounding boxes
        detected_image_url, detections = roi_service.detect_and_draw_boxes(image_bytes)


        return KneeDetectionResponse(
            filename=filename,
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
