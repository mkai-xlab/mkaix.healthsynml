"""Small request helpers shared by versioned API endpoints."""

from fastapi import HTTPException, UploadFile


async def read_uploaded_image(file: UploadFile) -> tuple[str, bytes]:
    """Validate an uploaded image once and return a stable filename and bytes.
    Args:
        file: An uploaded image file in PNG or JPEG format.
    Returns:
        A tuple containing the filename and the image bytes.
    Raises:
        HTTPException: If the uploaded file is invalid or if an error occurs during reading.
    """

    if file is None:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    # read the uploaded file and check if it's empty
    image_bytes = await file.read()
    
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    return file.filename or "uploaded-image", image_bytes
