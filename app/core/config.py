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
    MODEL_MODE: str = normalize_model_mode(
        os.getenv("MODEL_MODE", "densenet121")
    )
    MODEL_CHECKPOINT_PATH: str = os.getenv(
        "MODEL_CHECKPOINT_PATH",
        "checkpoints/densenet121/best_model.pth",
    )
    SE_RESNEXT_CHECKPOINT_PATH: str = os.getenv(
        "SE_RESNEXT_CHECKPOINT_PATH",
        "checkpoints/se_resnext50_32x4d/best_model (1).pth",
    )
    EFFICIENTNET_B0_CHECKPOINT_PATH: str = os.getenv(
        "EFFICIENTNET_B0_CHECKPOINT_PATH",
        "checkpoints/efficientnet_b0/best_model.pth",
    )
    YOLO_CHECKPOINT_PATH: str = os.getenv(
        "YOLO_CHECKPOINT_PATH", "checkpoints/yolov8/best.pt"
    )
    IMG_SIZE: int = int(os.getenv("IMG_SIZE", "384"))


settings = Settings()
