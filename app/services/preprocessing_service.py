import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from app.core.config import settings

class SquarePadOpenCV(object):
    """Pads a rectangular X-ray image to a square, preserving the aspect ratio of the joint space."""
    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        max_wh = max(h, w)
        pad_top = (max_wh - h) // 2
        pad_bottom = max_wh - h - pad_top
        pad_left = (max_wh - w) // 2
        pad_right = max_wh - w - pad_left
        
        padded_image = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right, 
            borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        return padded_image

class OpenCVCLAHE(object):
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance bone textures."""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(img_lab)
        clahe_l_channel = clahe.apply(l_channel)
        merged_lab_image = cv2.merge((clahe_l_channel, a_channel, b_channel))
        return cv2.cvtColor(merged_lab_image, cv2.COLOR_LAB2RGB)

class PreprocessingService:
    """Service to resize and normalize image input for deep learning models."""
    def __init__(self):
        self.img_size = settings.IMG_SIZE
        self.transform = transforms.Compose([
            SquarePadOpenCV(),
            OpenCVCLAHE(),
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_bytes: bytes) -> tuple[torch.Tensor, np.ndarray]:
        """
        Converts raw image bytes to a preprocessed tensor ready for inference.
        Also returns the original RGB image for use in Grad-CAM visualization.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image from bytes. Ensure file is a valid image (PNG/JPEG).")
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img_rgb)
        return tensor.unsqueeze(0), img_rgb

# Singleton instance
preprocessing_service = PreprocessingService()
