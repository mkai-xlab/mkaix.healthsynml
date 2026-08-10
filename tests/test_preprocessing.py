import cv2
import numpy as np

from app.services.preprocessing_service import (
    OpenCVCLAHE,
    SquarePadOpenCV,
    preprocessing_service,
)


def _encoded_asymmetric_image() -> bytes:
    image = np.zeros((180, 300, 3), dtype=np.uint8)
    image[:, :90] = (220, 40, 20)
    image[40:140, 180:280] = (20, 180, 240)
    success, buffer = cv2.imencode(".png", image)
    assert success
    return buffer.tobytes()


def test_inference_preprocessing_matches_checkpoint_dimensions():
    tensor, processed = preprocessing_service.preprocess_image(
        _encoded_asymmetric_image()
    )

    assert tensor.shape == (1, 3, 384, 384)
    assert processed.shape == (384, 384, 3)
    assert np.isfinite(tensor.numpy()).all()


def test_inference_uses_clahe_1_25_before_square_padding():
    operations = preprocessing_service.spatial_transform.transforms

    assert isinstance(operations[0], OpenCVCLAHE)
    assert operations[0].clip_limit == 1.25
    assert isinstance(operations[1], SquarePadOpenCV)
