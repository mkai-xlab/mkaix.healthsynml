# Application-wide configuration and environment variables loader
import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

class Settings:
    AWS_PROFILE: str = os.getenv("AWS_PROFILE", "duy")
    AWS_REGION: str = os.getenv("AWS_REGION", "ap-southeast-1")
    AWS_S3_MODELS_BUCKET: str = os.getenv(
        "AWS_S3_MODELS_BUCKET", 
        "healthsync-ml-models-dev-819109476069-ap-southeast-1-an"
    )
    
    DEFAULT_MODEL_NAME: str = os.getenv("DEFAULT_MODEL_NAME", "densenet201")
    MODEL_CHECKPOINT_PATH: str = os.getenv("MODEL_CHECKPOINT_PATH", "checkpoints/densenet201/best_model.pth")
    YOLO_CHECKPOINT_PATH: str = os.getenv("YOLO_CHECKPOINT_PATH", "checkpoints/yolov8/best.pt")
    IMG_SIZE: int = int(os.getenv("IMG_SIZE", "224"))
    ORDINAL_TYPE: str = os.getenv("ORDINAL_TYPE", "focal_corn")

settings = Settings()
