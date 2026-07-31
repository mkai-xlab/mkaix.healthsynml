import cv2
import numpy as np
import pytest

from app.services.roi_service import (
    NO_KNEE_ROI_MESSAGE,
    YOLO_ROI_EXPANSION,
    ROIService,
    make_square_roi,
)


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
    ["detect_and_draw_boxes", "detect_knees_with_coords"],
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


def test_make_square_roi_expands_the_larger_box_dimension():
    image = np.full((100, 200, 3), 255, dtype=np.uint8)

    crop = make_square_roi(image, [50, 10, 130, 90])

    expected_side = int(np.ceil(80 * YOLO_ROI_EXPANSION))
    assert crop.shape == (expected_side, expected_side, 3)
    assert np.all(crop == 255)


def test_make_square_roi_pads_only_when_the_expansion_reaches_an_edge():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)

    crop = make_square_roi(image, [0, 0, 20, 40])

    expected_side = int(np.ceil(40 * YOLO_ROI_EXPANSION))
    assert crop.shape == (expected_side, expected_side, 3)
    assert np.all(crop[0, 0] == 0)
    assert np.any(crop == 255)
