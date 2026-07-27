import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from app.core.config import settings


class SquarePadOpenCV:
    """Pad a rectangular ROI to square without changing its aspect ratio."""

    def __call__(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        maximum = max(height, width)
        pad_top = (maximum - height) // 2
        pad_bottom = maximum - height - pad_top
        pad_left = (maximum - width) // 2
        pad_right = maximum - width - pad_left
        return cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )


class OpenCVCLAHE:
    """Apply the same LAB-space CLAHE transform used during training."""

    def __init__(self, clip_limit: float = 1.25, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )
        image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        lightness, channel_a, channel_b = cv2.split(image_lab)
        enhanced = clahe.apply(lightness)
        return cv2.cvtColor(
            cv2.merge((enhanced, channel_a, channel_b)), cv2.COLOR_LAB2RGB
        )


def canonicalize_knee_laterality(
    image_rgb: np.ndarray, knee_side: str
) -> tuple[np.ndarray, bool]:
    """Retain natural laterality for the non-canonical production checkpoint."""
    return image_rgb, False


class PreprocessingService:
    """Decode and reproduce the checkpoint's deterministic inference transform."""

    def __init__(self):
        self.img_size = settings.IMG_SIZE
        self.crop_size = settings.CROP_SIZE
        self.spatial_transform = transforms.Compose(
            [
                OpenCVCLAHE(clip_limit=1.25),
                SquarePadOpenCV(),
                transforms.ToPILImage(),
                transforms.Resize((self.img_size, self.img_size)),
            ]
        )
        self.tensor_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def preprocess_image(
        self, image_bytes: bytes, knee_side: str = "unknown"
    ) -> tuple[torch.Tensor, np.ndarray, bool]:
        """Return the model tensor and the exactly aligned display image."""
        encoded = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(
                "Could not decode image. Upload a valid PNG or JPEG image."
            )

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb, was_mirrored = canonicalize_knee_laterality(
            image_rgb, knee_side
        )
        processed_image = np.array(
            self.spatial_transform(image_rgb), dtype=np.uint8, copy=True
        )
        tensor = self.tensor_transform(processed_image).unsqueeze(0)
        return tensor, processed_image, was_mirrored


preprocessing_service = PreprocessingService()
