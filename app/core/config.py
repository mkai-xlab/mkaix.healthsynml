import os

from dotenv import load_dotenv


load_dotenv()


MODEL_MODE_ALIASES = {
    "densenet121": "densenet121",
    "dense_net_121": "densenet121",
    "se_resnext": "se_resnext",
    "seresnext50_32x4d": "se_resnext",
    "se_resnext50_32x4d": "se_resnext",
    "efficientnet": "efficientnet_b0",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet0": "efficientnet_b0",
    "ensemble": "ensemble",
}


def normalize_model_mode(value: str) -> str:
    normalized = value.strip().lower()
    try:
        return MODEL_MODE_ALIASES[normalized]
    except KeyError as error:
        allowed = "densenet121, se_resnext, efficientnet_b0, ensemble"
        raise ValueError(
            f"Unsupported MODEL_MODE={value!r}. Allowed values: {allowed}."
        ) from error


class Settings:
    AWS_PROFILE: str = os.getenv("AWS_PROFILE", "duy")
    AWS_REGION: str = os.getenv("AWS_REGION", "ap-southeast-1")
    AWS_S3_MODELS_BUCKET: str = os.getenv(
        "AWS_S3_MODELS_BUCKET",
        "healthsync-ml-models-dev-819109476069-ap-southeast-1-an",
    )

    MODEL_MODE: str = normalize_model_mode(
        os.getenv("MODEL_MODE", "densenet121")
    )
    # Retained for modules outside the inference pipeline that still read it.
    DEFAULT_MODEL_NAME: str = MODEL_MODE
    MODEL_CHECKPOINT_PATH: str = os.getenv(
        "MODEL_CHECKPOINT_PATH",
        "checkpoints/densenet121/best_model.pth",
    )
    EXPECTED_MODEL_ARCHITECTURE: str = os.getenv(
        "EXPECTED_MODEL_ARCHITECTURE",
        "timm_densenet121_linear_gradcam",
    )
    SE_RESNEXT_CHECKPOINT_PATH: str = os.getenv(
        "SE_RESNEXT_CHECKPOINT_PATH",
        "checkpoints/se_resnext50_32x4d/best_model (1).pth",
    )
    EXPECTED_SE_RESNEXT_ARCHITECTURE: str = os.getenv(
        "EXPECTED_SE_RESNEXT_ARCHITECTURE",
        "final_native_cam_ce",
    )
    EFFICIENTNET_B0_CHECKPOINT_PATH: str = os.getenv(
        "EFFICIENTNET_B0_CHECKPOINT_PATH",
        "checkpoints/efficientnet_b0/best_model.pth",
    )
    EXPECTED_EFFICIENTNET_B0_ARCHITECTURE: str = os.getenv(
        "EXPECTED_EFFICIENTNET_B0_ARCHITECTURE",
        "efficientnet_b0_final_native_cam_ce",
    )
    ENSEMBLE_DENSENET_WEIGHT: float = float(
        os.getenv("ENSEMBLE_DENSENET_WEIGHT", "0.55")
    )
    ENSEMBLE_SE_RESNEXT_WEIGHT: float = float(
        os.getenv("ENSEMBLE_SE_RESNEXT_WEIGHT", "0.45")
    )
    ENSEMBLE_EFFICIENTNET_B0_WEIGHT: float = float(
        os.getenv("ENSEMBLE_EFFICIENTNET_B0_WEIGHT", "0.00")
    )
    YOLO_CHECKPOINT_PATH: str = os.getenv(
        "YOLO_CHECKPOINT_PATH", "checkpoints/yolov8/best.pt"
    )
    IMG_SIZE: int = int(os.getenv("IMG_SIZE", "384"))
    CROP_SIZE: int = int(os.getenv("CROP_SIZE", "384"))
    ORDINAL_TYPE: str = os.getenv("ORDINAL_TYPE", "ce")


settings = Settings()
