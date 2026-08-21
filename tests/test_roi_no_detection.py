"""
Tests for app.services.roi_service.

Functions under test
--------------------
  assign_knee_sides(knees, image_width)
      Input  : list of knee dicts, each containing {"box": [x1, y1, x2, y2]}
               and the pixel width of the full image
      Output : (ordered_knees, sides)
                   ordered_knees — knees sorted left → right
                   sides         — ["right", "left"] or ["right"] or ["left"]

  make_square_roi(image, box)
      Input  : full image numpy array, YOLO box [x1, y1, x2, y2]
      Output : square numpy crop of shape (side, side, 3)
               side = ceil(max(w, h) * YOLO_ROI_EXPANSION)

  ROIService.detect_and_draw_boxes(image_bytes)  /  detect_knees_with_coords(image_bytes)
      Input  : raw PNG bytes of a full X-ray
      Output : (ROI images as bytes, knee coordinates)
               Both raise ValueError(NO_KNEE_ROI_MESSAGE) when YOLO finds nothing.
"""
import cv2
import numpy as np
import pytest

from app.services.roi_service import (
    NO_KNEE_ROI_MESSAGE,
    YOLO_ROI_EXPANSION,
    ROIService,
    assign_knee_sides,
    make_square_roi,
)


# ---------------------------------------------------------------------------
# Mock helpers — replace the real YOLO detector with one that finds nothing.
# ---------------------------------------------------------------------------
class _EmptyResult:
    """Simulates one YOLO result object with zero detected boxes."""
    boxes = []


class _EmptyDetector:
    """
    Simulates a YOLO model whose predict() always returns a result with no boxes.
    Used to force the "no knee detected" code path.
    """
    def predict(self, **kwargs):
        return [_EmptyResult()]


# ---------------------------------------------------------------------------
# Helper: synthetic grayscale PNG bytes (128×192, all zeros).
# ---------------------------------------------------------------------------
def _image_bytes() -> bytes:
    image = np.zeros((128, 192, 3), dtype=np.uint8)
    success, buffer = cv2.imencode(".png", image)
    assert success
    return buffer.tobytes()


# ---------------------------------------------------------------------------
# Helper: builds a minimal ROIService whose .model is the mock detector.
# ---------------------------------------------------------------------------
def _service() -> ROIService:
    # __new__ bypasses __init__ so we can inject our mock without a real model file
    service = ROIService.__new__(ROIService)
    service.model = _EmptyDetector()
    return service


# ---------------------------------------------------------------------------
# Test: assign_knee_sides — two knees, left and right
# ---------------------------------------------------------------------------
def test_assign_knee_sides_orders_two_knees_left_to_right():
    """
    Input  : two knee boxes on a 1000px-wide image
               knee[0]: box [600, 20, 800, 220]  → centre x = 700  (right side)
               knee[1]: box [ 50, 20, 250, 220]  → centre x = 150  (left side)
    Expected output
             ordered_knees : [knees[1], knees[0]]   ← left-first order
             sides         : ["right", "left"]
    """
    knees = [{"box": [600, 20, 800, 220]}, {"box": [50, 20, 250, 220]}]

    ordered_knees, sides = assign_knee_sides(knees, image_width=1000)

    assert ordered_knees == [knees[1], knees[0]]  # left knee first
    assert sides == ["right", "left"]              # radiographic convention


# ---------------------------------------------------------------------------
# Test: assign_knee_sides — one knee, position determines side
# ---------------------------------------------------------------------------
def test_assign_knee_sides_uses_position_for_one_knee():
    """
    Input  : one knee box [50, 20, 250, 220] on a 1000px-wide image
               centre x = 150  (left half of image → RIGHT knee)
    Expected output: sides = ["right"]
    """
    knees = [{"box": [50, 20, 250, 220]}]

    _, sides = assign_knee_sides(knees, image_width=1000)

    assert sides == ["right"]


# ---------------------------------------------------------------------------
# Test: both ROI methods raise when YOLO detects nothing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "method_name",
    ["detect_and_draw_boxes", "detect_knees_with_coords"],
)
def test_roi_methods_reject_images_without_a_detection(method_name):
    """
    Input  : synthetic PNG with no real knee in it
             The mock _EmptyDetector returns zero boxes.

    Expected output
             ValueError whose message is exactly NO_KNEE_ROI_MESSAGE
    """
    with pytest.raises(ValueError, match="No knee joint ROI was detected") as error:
        getattr(_service(), method_name)(_image_bytes())

    assert str(error.value) == NO_KNEE_ROI_MESSAGE


# ---------------------------------------------------------------------------
# Test: ROI methods reject a service whose detector was never loaded
# ---------------------------------------------------------------------------
def test_roi_methods_reject_an_unavailable_detector():
    """
    Input  : ROIService whose .model is None (detector not loaded)

    Expected output
             RuntimeError whose message is "YOLO ROI detector is unavailable"
    """
    service = ROIService.__new__(ROIService)
    service.model = None

    with pytest.raises(RuntimeError, match="YOLO ROI detector is unavailable"):
        service.detect_knees_with_coords(_image_bytes())


# ---------------------------------------------------------------------------
# Test: make_square_roi — box is fully inside image (no padding needed)
# ---------------------------------------------------------------------------
def test_make_square_roi_expands_the_larger_box_dimension():
    """
    Input  : white 100×200 image, box [50, 10, 130, 90]
               width  = 130 - 50  = 80
               height = 90  - 10  = 80
               larger dimension = 80
               expected side    = ceil(80 × 1.15) = 92

    Expected output
             crop shape : (92, 92, 3)
             all pixels : 255 (white, original colour is preserved)
    """
    image = np.full((100, 200, 3), 255, dtype=np.uint8)

    crop = make_square_roi(image, [50, 10, 130, 90])

    expected_side = int(np.ceil(80 * YOLO_ROI_EXPANSION))
    assert crop.shape == (expected_side, expected_side, 3)
    assert np.all(crop == 255)


# ---------------------------------------------------------------------------
# Test: make_square_roi — expansion reaches image edge, padding appears
# ---------------------------------------------------------------------------
def test_make_square_roi_pads_only_when_the_expansion_reaches_an_edge():
    """
    Input  : 100×100 white image, box [0, 0, 20, 40] at top-left corner
               height = 40
               expanded side = ceil(40 × 1.15) = 46
               The box starts at y=0 and x=0, so the expanded crop extends
               beyond the image boundary at the top and left edges.

    Expected output
             crop shape : (46, 46, 3)
             top-left corner pixel (padded area)  : 0   (black, padding)
             somewhere in the image area          : 255 (white, original content)
    """
    image = np.full((100, 100, 3), 255, dtype=np.uint8)

    crop = make_square_roi(image, [0, 0, 20, 40])

    expected_side = int(np.ceil(40 * YOLO_ROI_EXPANSION))
    assert crop.shape == (expected_side, expected_side, 3)
    # Padded area (outside the original image) is filled with black (0, 0, 0)
    assert np.all(crop[0, 0] == 0)
    # Some pixel in the crop still has the original white value
    assert np.any(crop == 255)
