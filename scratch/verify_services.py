import torch
import numpy as np
from app.services.preprocessing_service import preprocessing_service
from app.services.inference_service import inference_service
from app.services.gradcam_service import gradcam_service
from app.ml.pipelines.knee_oa_pipeline import KneeOAPipeline

print("Importing all services succeeded!")

# Instantiate a mock image (a simple solid gray image of 300x400)
import cv2
img = (np.ones((300, 400, 3)) * 128).astype(np.uint8)
_, buffer = cv2.imencode(".jpg", img)
image_bytes = buffer.tobytes()

print("Preprocessing mock image bytes...")
tensor, img_rgb = preprocessing_service.preprocess_image(image_bytes)
print(f"Preprocessed tensor shape: {tensor.shape}")
print(f"Original RGB image shape: {img_rgb.shape}")

print("Initializing pipeline...")
pipeline = KneeOAPipeline()
print("Pipeline initialized successfully!")

# Let's run postprocess with dummy logits
print("Testing postprocess...")
dummy_logits = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
result = pipeline.postprocess(dummy_logits)
print(f"Postprocess output: {result}")

print("All verifications passed successfully!")
