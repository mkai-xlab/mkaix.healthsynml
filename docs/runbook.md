# 📖 Operations Runbook

This runbook describes configurations, troubleshooting steps, and error resolving procedures.

---

## ⚙️ Environment Variables Config

The following environment variables can be configured in the `.env` file or exported to the shell:

| Variable Name | Description | Default Value | Allowed Values |
| :--- | :--- | :--- | :--- |
| `APP_NAME` | Display name of the API | `"MKAI Knee Osteoarthritis API"` | Any string |
| `APP_ENV` | Running environment mode | `development` | `development`, `production`, `testing` |
| `LOG_LEVEL` | Logging filter verbosity | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `IMAGE_SIZE` | Model input dimensions | `224` | Integers (e.g., `224`, `256`, `512`) |
| `MODEL_WEIGHTS_DIR` | Directory containing keras weights | `model_weights/` | Relative or absolute path |

---

## 🚨 Error Codes & REST Responses

Standard API exceptions return a structured error response matching the `ErrorResponse` schema (HTTP status codes 400, 422, or 500):

```json
{
  "error": {
    "code": "INVALID_INPUT_FILE_ERROR",
    "message": "Uploaded file is empty.",
    "details": {}
  }
}
```

### Common Error Classifications:

* **`INVALID_INPUT_FILE_ERROR` (HTTP 400)**:
  * *Trigger*: The uploaded file size is 0 bytes, or the extension does not match permitted image formats (`.png`, `.jpg`, `.jpeg`) or DICOM configurations.
  * *Fix*: Verify the format of the selected file before upload.
* **`DICOM_PROCESSING_ERROR` (HTTP 422)**:
  * *Trigger*: The byte stream is corrupted, or does not contain `PixelData` tags.
  * *Fix*: Inspect the DICOM file using metadata viewers to ensure it is not corrupted and is uncompressed.
* **`MODEL_LOAD_ERROR` (HTTP 500)**:
  * *Trigger*: A model registry weight file load operation failed.
  * *Fix*: Verify the weights folder permissions and file paths.

---

## 📊 Troubleshooting Checklist

### ⚠️ Warning: Running in Mock Inference Mode
* **Symptoms**: Predictions run immediately, but logs display warnings: `Weights file not found for efficientnet_b0... Falling back to Mock Inference Mode`.
* **Fix**: Ensure that the three model weight files are downloaded and placed under the respective folders in the project directory:
  * `model_weights/efficientnet_b0/best_model.keras`
  * `model_weights/densenet121/best_model.keras`
  * `model_weights/mobilenet_v2/best_model.keras`

### 🐌 Performance Bottlenecks (X-Process-Time is high)
* **Symptoms**: The custom header `X-Process-Time` returns values greater than 500ms.
* **Checks**:
  1. Open logs to look at execution time breakdowns.
  2. If `inference_ms` is high, check if TensorFlow is running on GPU:
     ```bash
     python -c "import tensorflow as tf; print('GPUs Available:', tf.config.list_physical_devices('GPU'))"
     ```
  3. If `dicom_processing_ms` is high, ensure the uploaded files are not excessively large (DICOM files containing hundreds of slices). The API is optimized for single-slice radiographs.
