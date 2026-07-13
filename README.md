# MKAI Knee Osteoarthritis API (mkai_x_knee_oa_api)

This repository contains the REST API package schema designed for predicting the **Kellgren-Lawrence Grade (0 to 4)** of knee osteoarthritis from radiographic images (DICOM/PNG/JPEG). The API is built using **FastAPI** and hosts an ensemble pipeline consisting of three deep learning models: **EfficientNet-B0**, **DenseNet-121**, and **MobileNet-V2**.

---

## 📂 Package Directory Structure

```text
mkai_x_knee_oa_api/
│
├── app/
│   ├── main.py                     # FastAPI main entrypoint and middlewares
│   │
│   ├── api/
│   │   ├── dependencies.py         # Routing dependencies (e.g. pipeline instances)
│   │   │
│   │   └── v1/
│   │       ├── router.py           # Combined API V1 routers
│   │       │
│   │       └── endpoints/
│   │           ├── health.py       # API status & model readiness checking
│   │           ├── prediction.py   # Upload and predict knee OA grades
│   │           └── model_info.py   # Registered models specs & metadata
│   │
│   ├── core/
│   │   ├── config.py               # Settings loader (dotenv integration)
│   │   ├── constants.py            # Global constant lookup mapping
│   │   ├── exceptions.py           # Structured error exception subclasses
│   │   └── logging_config.py       # Custom log decorators and configuration
│   │
│   ├── schemas/
│   │   ├── prediction.py           # Pydantic schemas for endpoint outputs
│   │   ├── model_info.py           # Pydantic schemas for model attributes
│   │   └── error.py                # Standard error response wrapper formats
│   │
│   ├── services/
│   │   ├── dicom_service.py        # DICOM file byte decoding and metadata extraction
│   │   ├── preprocessing_service.py# Resizing and torchvision ImageNet normalization
│   │   ├── roi_service.py          # Knee joint extraction of region of interest
│   │   ├── inference_service.py    # Invocation of active registered networks
│   │   ├── ensemble_service.py     # Blended predictions average calculation
│   │   └── gradcam_service.py      # Grad-CAM heatmap visualization creation
│   │
│   ├── ml/
│   │   ├── model_registry.py       # Model wrapper loader & registry singleton
│   │   │
│   │   ├── models/
│   │   │   ├── base_model.py       # Abstract base class with Mock Inference fallback
│   │   │   ├── efficientnet_b0_model.py # Wrapper around EfficientNet-B0 network
│   │   │   ├── densenet121_model.py     # Wrapper around DenseNet-121 network
│   │   │   └── mobilenet_v2_model.py    # Wrapper around MobileNet-V2 network
│   │   │
│   │   └── pipelines/
│   │       └── knee_oa_pipeline.py # End-to-end processing & inference orchestrator
│   │
│   └── utils/
│       ├── file_utils.py           # Binary check, extension validation utilities
│       └── image_utils.py          # PIL/numpy array and Base64 format converters
│
├── docs/
│   ├── architecture.md             # Data flow diagram and components explanation
│   └── runbook.md                  # Deployment guides, monitoring metrics, error codes
│
├── model_weights/                  # Model weights (ignored in Git; populated locally)
│   ├── efficientnet_b0/
│   │   └── best_model.keras        # EfficientNet-B0 weights (.gitkeep placeholder)
│   ├── densenet121/
│   │   └── best_model.keras        # DenseNet-121 weights (.gitkeep placeholder)
│   └── mobilenet_v2/
│       └── best_model.keras        # MobileNet-V2 weights (.gitkeep placeholder)
│
├── notebooks/                      # Experimental/Google Colab notebooks
│   ├── knee_osteoarthritis_colab.ipynb
│   └── readdicom.ipynb
│
├── tests/                          # Pytest test suites
│   ├── test_health.py              # Health check endpoint tests
│   ├── test_dicom_service.py       # DICOM parsing checks
│   ├── test_preprocessing.py       # Image resizing and scaling checks
│   └── test_prediction_api.py      # Mock API upload diagnostics
│
├── requirements.txt                # List of system package requirements
├── .env                            # Environment variables config template
├── .gitignore                      # Git exclusion rules
└── README.md                       # This file
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Setup Environment
```bash
# Clone the repository and navigate to the package
cd mkai_x_knee_oa_api

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running local server
Start the local development server:
```bash
uvicorn app.main:app --reload
```
Once started, open:
* **Interactive API Documentation (Swagger)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 4. Running Test Suite
Execute the automated tests using:
```bash
pytest
```

---

## 🛠 ML Model Weights & Mock Inference Mode
To prevent runtime crashes when model weights are not present, **Mock Inference Mode** will activate automatically if any `.keras` weight files are missing in `model_weights/`. 
* Logs will warn about running in **Mock Mode**.
* Mock mode generates deterministic mock predictions based on image contents.
* To use real predictions, place your trained Keras models named `best_model.keras` inside their respective folders in `model_weights/`.
