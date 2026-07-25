FROM dhi.io/python:3@sha256:6b0b46d3451ae138084c8aea720b0cd458309540e656db66406065830305caef

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/home/nonroot/.local/bin:${PATH}"
ENV MODEL_MODE="densenet121"
ENV MODEL_CHECKPOINT_PATH="/app/checkpoints/densenet121/best_model.pth"
ENV EXPECTED_MODEL_ARCHITECTURE="final_linear_native_cam"
ENV SE_RESNEXT_CHECKPOINT_PATH="/app/checkpoints/se_resnext50_32x4d/best_model (1).pth"
ENV EXPECTED_SE_RESNEXT_ARCHITECTURE="final_native_cam_ce"
ENV EFFICIENTNET_B0_CHECKPOINT_PATH="/app/checkpoints/efficientnet_b0/best_model.pth"
ENV EXPECTED_EFFICIENTNET_B0_ARCHITECTURE="efficientnet_b0_final_native_cam_ce"
ENV ENSEMBLE_DENSENET_WEIGHT="0.55"
ENV ENSEMBLE_SE_RESNEXT_WEIGHT="0.45"
ENV ENSEMBLE_EFFICIENTNET_B0_WEIGHT="0.00"
ENV YOLO_CHECKPOINT_PATH="/app/checkpoints/yolov8/best.pt"
ENV IMG_SIZE="384"
ENV CROP_SIZE="384"
ENV ORDINAL_TYPE="ce"
ENV YOLO_CONFIG_DIR="/tmp/Ultralytics"

# Set the working directory in the container
WORKDIR /app

# Copy only requirements to cache them in docker layer
COPY requirements.txt .

# Install Python dependencies using exec form (without /bin/sh shell)
RUN ["python", "-m", "pip", "install", "--upgrade", "pip"]
RUN ["python", "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"]
RUN ["python", "-m", "pip", "install", "--no-cache-dir", "-r", "requirements.txt"]
RUN ["python", "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-python-headless"]
RUN ["python", "-m", "pip", "install", "--no-cache-dir", "opencv-python-headless"]

# Copy the rest of the application code to the container
COPY . .

# Ultralytics appends its own "Ultralytics" subdirectory to this parent.
ENV YOLO_CONFIG_DIR="/tmp"

# Expose the port the app runs on
EXPOSE 8005

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8005/api/v1/health', timeout=5)"]

# Command to run the application using Uvicorn (exec form)
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]
