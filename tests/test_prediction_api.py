"""Contract tests for the ``POST /api/v1/predict`` endpoint."""

import asyncio
import base64
import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from fastapi import HTTPException


class _Upload:
    """Minimal async upload object accepted by ``read_uploaded_image``."""

    def __init__(self, data: bytes, filename: str):
        self._data = data
        self.filename = filename

    async def read(self) -> bytes:
        return self._data


def _png_bytes() -> bytes:
    """Return a small valid PNG without requiring model or detector binaries."""
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    image[8:24, 12:36] = (180, 180, 180)
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


def _import_endpoint_with_fake_service(monkeypatch):
    """Import the endpoint while preventing production checkpoint loading."""
    fake_result = {
        "filename": "knee.png",
        "predictions": [
            {
                "predicted_class": 2,
                "predicted_grade": "2Mild",
                "confidence": 0.72,
                "description": "Grade 2: Definite osteophytes and possible joint space narrowing.",
                "details": {
                    "0Normal": 0.04,
                    "1Doubtful": 0.09,
                    "2Mild": 0.72,
                    "3Moderate": 0.10,
                    "4Severe": 0.05,
                },
                "box": [4, 5, 40, 28],
                "yolo_confidence": 0.93,
                "knee_side": "right",
                "roi_image": "data:image/png;base64," + base64.b64encode(b"roi").decode(),
                "gradcam_image": "data:image/png;base64," + base64.b64encode(b"cam").decode(),
            }
        ],
        "annotated_image": "data:image/jpeg;base64," + base64.b64encode(b"annotated").decode(),
    }
    fake_service = SimpleNamespace(predict_image=Mock(return_value=fake_result))
    fake_module = types.ModuleType("app.services.prediction_service")
    fake_module.prediction_service = fake_service
    monkeypatch.setitem(sys.modules, "app.services.prediction_service", fake_module)

    # The endpoint imports the sibling detect-roi route, so stub this module too
    # to avoid initializing YOLO while importing the test subject.
    fake_roi_module = types.ModuleType("app.services.roi_service")
    fake_roi_module.roi_service = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "app.services.roi_service", fake_roi_module)

    endpoint_module_name = "app.api.v1.endpoints.prediction"
    sys.modules.pop(endpoint_module_name, None)
    endpoint = importlib.import_module(endpoint_module_name)
    return endpoint, fake_service, fake_result


def test_prediction_route_returns_the_public_response_schema(monkeypatch):
    """A valid upload is passed to the service and returned by the route."""
    endpoint, fake_service, expected = _import_endpoint_with_fake_service(monkeypatch)
    image_bytes = _png_bytes()
    upload = _Upload(image_bytes, "knee.png")

    response = asyncio.run(endpoint.predict_knee_oa(upload))

    assert response.filename == "knee.png"
    assert len(response.predictions) == 1
    prediction = response.predictions[0]
    assert prediction.predicted_class in range(5)
    assert prediction.predicted_grade == "2Mild"
    assert prediction.confidence == pytest.approx(0.72)
    assert sum(prediction.details.values()) == pytest.approx(1.0)
    assert prediction.box == [4, 5, 40, 28]
    assert prediction.knee_side == "right"
    assert prediction.roi_image.startswith("data:image/png;base64,")
    assert prediction.gradcam_image.startswith("data:image/png;base64,")
    assert response.annotated_image.startswith("data:image/jpeg;base64,")
    fake_service.predict_image.assert_called_once_with("knee.png", image_bytes)
    assert expected["filename"] == response.filename


def test_prediction_route_rejects_empty_upload(monkeypatch):
    """An empty multipart file becomes the documented HTTP 400 error."""
    endpoint, fake_service, _ = _import_endpoint_with_fake_service(monkeypatch)
    upload = _Upload(b"", "empty.png")

    with pytest.raises(HTTPException) as error:
        asyncio.run(endpoint.predict_knee_oa(upload))

    assert error.value.status_code == 400
    assert error.value.detail == "Uploaded file is empty."
    fake_service.predict_image.assert_not_called()
