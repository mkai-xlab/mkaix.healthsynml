import cv2
import numpy as np

from app.services.preprocessing_service import (
    canonicalize_knee_laterality,
    preprocessing_service,
)


def _encoded_asymmetric_image() -> bytes:
    image = np.zeros((180, 300, 3), dtype=np.uint8)
    image[:, :90] = (220, 40, 20)
    image[40:140, 180:280] = (20, 180, 240)
    success, buffer = cv2.imencode(".png", image)
    assert success
    return buffer.tobytes()


def test_right_knee_canonicalization_is_horizontal_mirror():
    image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    left, left_was_mirrored = canonicalize_knee_laterality(image, "left")
    right, right_was_mirrored = canonicalize_knee_laterality(image, "right")

    assert not left_was_mirrored
    assert right_was_mirrored
    assert np.array_equal(left, image)
    assert np.array_equal(right, image[:, ::-1])
    assert right.flags["C_CONTIGUOUS"]


def test_inference_preprocessing_matches_checkpoint_dimensions():
    tensor, processed, was_mirrored = preprocessing_service.preprocess_image(
        _encoded_asymmetric_image(), knee_side="right"
    )

    assert tensor.shape == (1, 3, 384, 384)
    assert processed.shape == (384, 384, 3)
    assert was_mirrored
    assert np.isfinite(tensor.numpy()).all()
