FROM dhi.io/python:3

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/home/nonroot/.local/bin:${PATH}"

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

# Expose the port the app runs on
EXPOSE 8005

# Command to run the application using Uvicorn (exec form)
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]
