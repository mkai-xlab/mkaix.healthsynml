# Unit Tests

Run the local suite:

```bash
make unit-test
```

This runs:

```bash
python -m pytest -q --disable-warnings --maxfail=1
```

## What Is Tested

| Test file | Function or behavior | Input | Expected output |
| --- | --- | --- | --- |
| `test_ensemble.py` | `weighted_soft_vote` | Two `[1, 5]` logit tensors and weights `0.55`, `0.45` | Weighted sum of each model's softmax probabilities; five probabilities sum to `1`. |
| `test_ensemble.py` | `weighted_soft_vote` validation | One model only | `ValueError`, because soft voting needs at least two models. |
| `test_ensemble.py` | `select_heatmap_component` | Two model probability tensors and one predicted class | The model name with the highest probability for that class. |
| `test_preprocessing.py` | `preprocess_image` | Asymmetric `180 x 300` PNG image | Tensor shape `[1, 3, 384, 384]`, display image shape `[384, 384, 3]`, and finite tensor values. |
| `test_preprocessing.py` | Transform order | Configured preprocessing transform | CLAHE with `clip_limit=1.25` runs before square padding. |
| `test_roi_no_detection.py` | `assign_knee_sides` | Two unsorted boxes or one left-side box | Two knees are sorted then labeled `right`, `left`; one left-side knee is labeled `right`. |
| `test_roi_no_detection.py` | ROI detection failures | PNG image with a fake detector that returns no boxes | Both ROI methods raise the standard no-ROI `ValueError`. |
| `test_roi_no_detection.py` | Detector availability | `ROIService` with no loaded detector | `RuntimeError` stating the detector is unavailable. |
| `test_roi_no_detection.py` | `make_square_roi` | Valid box fully inside an image | Square ROI with side `ceil(max(box width, box height) * 1.15)` and unchanged source pixels. |
| `test_roi_no_detection.py` | Edge padding | Box that expands beyond the upper-left image edge | Correct square size, black padding at the edge, and original pixels remain in the crop. |
| `test_model_mode.py` | `normalize_model_mode` | Supported names and aliases such as `dense_net_121` | Canonical mode: `densenet121`, `se_resnext`, or `ensemble`. |
| `test_model_mode.py` | Invalid model mode | `automatic` | `ValueError` containing `Unsupported MODEL_MODE`. |
| `test_gradcam_service.py` | `GradCAMService.extract_gradcam` | Small deterministic CNN and random `[1, 3, 32, 32]` image | Finite normalized `32 x 32` Grad-CAM map with values from `0` to `1`. |
| `test_gradcam_model.py` | Real checkpoint Grad-CAM | Mounted DenseNet or SE-ResNeXt checkpoint and zero `[1, 3, 384, 384]` image | Checkpoint architecture matches, weights load, and normalized Grad-CAM has image dimensions. Skips when checkpoints are absent. |

## Not Implemented Yet

These test files currently contain placeholders only and do not verify behavior:

- `test_health.py`: health endpoint response.
- `test_prediction_api.py`: prediction upload endpoint.
- `test_dicom_service.py`: DICOM parsing.

Add real input data, mocks, and assertions before relying on these files as coverage.
