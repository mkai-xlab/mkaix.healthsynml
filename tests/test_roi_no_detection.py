import cv2
import numpy as np
import pytest

from app.services.roi_service import NO_KNEE_ROI_MESSAGE, ROIService


class _EmptyResult:
    boxes = []


class _EmptyDetector:
    def predict(self, **kwargs):
        return [_EmptyResult()]


def _image_bytes() -> bytes:
    image = np.zeros((128, 192, 3), dtype=np.uint8)
    success, buffer = cv2.imencode(".png", image)
    assert success
    return buffer.tobytes()


def _service() -> ROIService:
    service = ROIService.__new__(ROIService)
    service.model = _EmptyDetector()
    return service


@pytest.mark.parametrize(
    "method_name",
    ["detect_and_draw_boxes", "crop_knees", "detect_knees_with_coords"],
)
def test_roi_methods_reject_images_without_a_detection(method_name):
    with pytest.raises(ValueError, match="No knee joint ROI was detected") as error:
        getattr(_service(), method_name)(_image_bytes())

    assert str(error.value) == NO_KNEE_ROI_MESSAGE


def test_roi_methods_reject_an_unavailable_detector():
    service = ROIService.__new__(ROIService)
    service.model = None

    with pytest.raises(RuntimeError, match="YOLO ROI detector is unavailable"):
        service.detect_knees_with_coords(_image_bytes())
