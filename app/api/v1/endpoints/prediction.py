from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.services.prediction_service import prediction_service
from app.services.roi_service import roi_service

router = APIRouter()

@router.post("", response_model=dict, status_code=status.HTTP_200_OK)
async def predict_knee_oa(file: UploadFile = File(..., description="Knee X-ray PNG/JPEG or DICOM file")):
    """
    Accepts an uploaded knee X-ray image file and processes it through the 
    service layer, preprocessing pipeline, and model registry for diagnosis.
    
    Returns:
        dict: A dictionary containing the filename, predicted KL grade, 
              grade description, confidence score, and class-wise probabilities.
    """
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="No file was uploaded."
        )
        
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Uploaded file is empty."
            )
            
        # Run prediction through prediction service
        result = prediction_service.predict_image(file.filename, image_bytes)
        return result
        
    except ValueError as e:
        # Catch specific preprocessing or value errors (e.g., failed to decode image)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=str(e)
        )
    except Exception as e:
        # Catch general runtime errors during inference
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Inference pipeline error: {str(e)}"
        )

@router.post("/detect-roi", response_model=dict, status_code=status.HTTP_200_OK)
async def detect_knee_roi(file: UploadFile = File(..., description="Knee X-ray PNG/JPEG/DICOM file")):
    """
    Accepts an uploaded knee X-ray image and runs YOLOv8 knee joint detection.
    Returns the base64-encoded image with bounding boxes drawn for validation.
    """
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="No file was uploaded."
        )
        
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Uploaded file is empty."
            )
            
        detected_image_url = roi_service.detect_and_draw_boxes(image_bytes)
        return {
            "filename": file.filename,
            "detected_image": detected_image_url
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"ROI detection error: {str(e)}"
        )


