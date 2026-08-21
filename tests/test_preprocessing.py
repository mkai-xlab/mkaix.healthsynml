"""
Tests for app.services.preprocessing_service.

Purpose
-------
The preprocessing pipeline converts a raw X-ray PNG (any aspect ratio,
any original size) into a fixed-size (3, 384, 384) tensor that a trained
classification model expects as input.

The pipeline (in order) is:
  1. CLAHE with clip_limit=1.25  — enhances contrast locally on each tile
  2. SquarePad                  — pads the image to a square with black pixels
  3. Resize (384, 384)           — scales the padded square to the target size
  4. Normalize                   — subtracts ImageNet mean, divides by std
  5. ToTensor                    — converts HWC → CHW float32
  6. Unsqueeze (dim=0)          — adds a batch dimension → (1, 3, 384, 384)

Input
-----
  A PNG image encoded as bytes (e.g. the body of a multipart/form-data POST).

Expected output
---------------
  - preprocessing_service.preprocess_image(bytes) returns (tensor, processed_img)
      tensor       — torch.Tensor of shape (1, 3, 384, 384), all values finite
      processed    — numpy array of shape (384, 384, 3), the image before
                     tensor conversion (useful for visualisation)

  - The first two transforms in spatial_transform must be
      (OpenCVCLAHE(clip_limit=1.25), SquarePadOpenCV)
"""
import cv2
import numpy as np

from app.services.preprocessing_service import (
    OpenCVCLAHE,
    SquarePadOpenCV,
    preprocessing_service,
)


# ---------------------------------------------------------------------------
# Helper: builds a synthetic asymmetric PNG with two coloured rectangles.
# ---------------------------------------------------------------------------
# Input  : none (generates data internally)
# Output : PNG bytes (height=180, width=300, asymmetric so SquarePad is tested)
#
# Why asymmetric?
#   Width (300) > height (180).  The pipeline must pad to a 300×300 square,
#   then resize to 384×384.  An all-zero image would give all-black after
#   padding, masking bugs in the pad implementation.
# ---------------------------------------------------------------------------
def _encoded_asymmetric_image() -> bytes:
    image = np.zeros((180, 300, 3), dtype=np.uint8)
    # Left half: reddish block
    image[:, :90] = (220, 40, 20)
    # Right-centre: cyan block — ensures padding area stays black
    image[40:140, 180:280] = (20, 180, 240)
    success, buffer = cv2.imencode(".png", image)
    assert success
    return buffer.tobytes()


def test_inference_preprocessing_matches_checkpoint_dimensions():
    """
    Input  : synthetic PNG bytes produced by _encoded_asymmetric_image()
    Output : (tensor, processed_img)

    Checks
      - tensor shape   : (1, 3, 384, 384)  ← what the model expects
      - processed shape: (384, 384, 3)      ← what Grad-CAM / CAM sees
      - all tensor values are finite (no NaN / Inf from normalisation)
    """
    tensor, processed = preprocessing_service.preprocess_image(
        _encoded_asymmetric_image()
    )

    assert tensor.shape == (1, 3, 384, 384)
    assert processed.shape == (384, 384, 3)
    assert np.isfinite(tensor.numpy()).all()


def test_inference_uses_clahe_1_25_before_square_padding():
    """
    Input  : none (reads the transform pipeline directly)
    Output : raises AssertionError if the pipeline order is wrong

    The API contract requires CLAHE (clip_limit=1.25) to run BEFORE
    SquarePad so that contrast enhancement applies to the full aspect ratio.
    Transforms are applied left-to-right, so:
      - operations[0] must be OpenCVCLAHE with clip_limit==1.25
      - operations[1] must be SquarePadOpenCV
    """
    operations = preprocessing_service.spatial_transform.transforms

    assert isinstance(operations[0], OpenCVCLAHE)
    assert operations[0].clip_limit == 1.25
    assert isinstance(operations[1], SquarePadOpenCV)
