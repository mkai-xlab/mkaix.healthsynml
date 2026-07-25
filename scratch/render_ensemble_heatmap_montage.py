"""Render a small ROI/native-CAM montage from the running smoke-test API."""

import base64
import json
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from smoke_test_ensemble_api import multipart_body


API_URL = os.getenv(
    "API_URL", "http://127.0.0.1:8005/api/v1/predict"
)
IMAGE_DIR = Path(os.getenv("IMAGE_DIR", "/test_images"))
DEFAULT_IMAGE_NAMES = (
    "9003175_20050511_00771504_png.rf.IoxlFMVl0YwSkDAlE75n.png",
    "9003430_20050602_00834204_png.rf.Kpd9DSkhjuW0yLXWtExz.png",
    "9063928_20050706_00936604_png.rf.LRfuXW5YKk9oloWTXpP5.png",
    "9066155_20050708_00966103_png.rf.MPbUeDHCeJ08c8TNVNVc.png",
)
IMAGE_NAMES = tuple(
    name.strip()
    for name in os.getenv("IMAGE_NAMES", ",".join(DEFAULT_IMAGE_NAMES)).split(",")
    if name.strip()
)


def decode_data_url(value: str) -> np.ndarray:
    raw = base64.b64decode(value.split(",", 1)[1])
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Could not decode API image")
    return image


def tile(image: np.ndarray, title: str) -> np.ndarray:
    image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)
    cv2.rectangle(image, (0, 0), (384, 34), (0, 0, 0), -1)
    cv2.putText(
        image,
        title,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def main() -> None:
    rows = []
    for image_name in IMAGE_NAMES:
        image_path = IMAGE_DIR / image_name
        body, content_type = multipart_body(image_path)
        request = urllib.request.Request(
            API_URL,
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read())
        for index, prediction in enumerate(payload["predictions"]):
            roi = decode_data_url(prediction["roi_image"])
            cam = decode_data_url(prediction["gradcam_image"])
            side = prediction["knee_side"]
            grade = prediction["predicted_class"]
            confidence = prediction["confidence"]
            short_name = image_name.split("_", 1)[0]
            rows.append(
                np.hstack(
                    [
                        tile(roi, f"{short_name} {side} ROI"),
                        tile(cam, f"Grade {grade}, confidence {confidence:.3f}"),
                    ]
                )
            )
    montage = np.vstack(rows)
    output = os.getenv(
        "OUTPUT_PATH", "/tmp/ensemble_heatmap_montage.jpg"
    )
    if not cv2.imwrite(output, montage):
        raise RuntimeError("Could not write heatmap montage")
    print(output)


if __name__ == "__main__":
    main()
