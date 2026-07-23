import os

from dotenv import load_dotenv


load_dotenv()


class Settings:
    AWS_PROFILE: str = os.getenv("AWS_PROFILE", "duy")
    AWS_REGION: str = os.getenv("AWS_REGION", "ap-southeast-1")
    AWS_S3_MODELS_BUCKET: str = os.getenv(
        "AWS_S3_MODELS_BUCKET",
        "healthsync-ml-models-dev-819109476069-ap-southeast-1-an",
    )

    DEFAULT_MODEL_NAME: str = os.getenv("DEFAULT_MODEL_NAME", "densenet121")
    MODEL_CHECKPOINT_PATH: str = os.getenv(
        "MODEL_CHECKPOINT_PATH",
        "checkpoints/densenet121/best_model.pth",
    )
    EXPECTED_MODEL_ARCHITECTURE: str = os.getenv(
        "EXPECTED_MODEL_ARCHITECTURE",
        "canonical_final_linear_cam",
    )
    SE_RESNEXT_CHECKPOINT_PATH: str = os.getenv(
        "SE_RESNEXT_CHECKPOINT_PATH",
        "checkpoints/se_resnext50_32x4d/best_model (1).pth",
    )
    EXPECTED_SE_RESNEXT_ARCHITECTURE: str = os.getenv(
        "EXPECTED_SE_RESNEXT_ARCHITECTURE",
        "final_native_cam_ce",
    )
    YOLO_CHECKPOINT_PATH: str = os.getenv(
        "YOLO_CHECKPOINT_PATH", "checkpoints/yolov8/best.pt"
    )
    IMG_SIZE: int = int(os.getenv("IMG_SIZE", "400"))
    CROP_SIZE: int = int(os.getenv("CROP_SIZE", "384"))
    ORDINAL_TYPE: str = os.getenv("ORDINAL_TYPE", "ce")


settings = Settings()
