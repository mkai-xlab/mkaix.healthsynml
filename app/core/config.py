import os

from dotenv import load_dotenv

# load environment variables from .env file if it exists
load_dotenv()

# Environment is intentionally limited to runtime choices and checkpoint paths.
# Training-only settings do not belong in this inference service.
MODEL_MODE_ALIASES = {
    "densenet121": "densenet121",
    "dense_net_121": "densenet121",
    "se_resnext": "se_resnext",
    "seresnext50_32x4d": "se_resnext",
    "se_resnext50_32x4d": "se_resnext",
    "ensemble": "ensemble",
}


def normalize_model_mode(value: str) -> str:
    """Normalize the model mode string to a canonical form.
    Args:
        value: The model mode string to normalize.
    Returns:
        The normalized model mode string.
    Raises:
        ValueError: If the model mode is not supported.
    """

    normalized = value.strip().lower()
    try:
        return MODEL_MODE_ALIASES[normalized]
    except KeyError as error:
        allowed = "densenet121, se_resnext, ensemble"
        raise ValueError(
            f"Unsupported MODEL_MODE={value!r}. Allowed values: {allowed}."
        ) from error


class Settings:
    """Runtime settings read once when the process starts."""

    MODEL_MODE: str = normalize_model_mode(
        os.getenv("MODEL_MODE", "densenet121")
    )
    DENSENET121_CHECKPOINT_PATH: str = os.getenv(
        "DENSENET121_CHECKPOINT_PATH",
        "checkpoints/densenet121/best_model.pth",
    )
    SE_RESNEXT_CHECKPOINT_PATH: str = os.getenv(
        "SE_RESNEXT_CHECKPOINT_PATH",
        (
            "checkpoints/se_resnext50_32x4d/"
            "2026-08-08_08-35-38_UTC_linear_gradcam/best_model.pth"
        ),
    )
    YOLO_CHECKPOINT_PATH: str = os.getenv(
        "YOLO_CHECKPOINT_PATH", "checkpoints/yolov8/best.pt"
    )
    IMG_SIZE: int = int(os.getenv("IMG_SIZE", "384"))


settings = Settings()
