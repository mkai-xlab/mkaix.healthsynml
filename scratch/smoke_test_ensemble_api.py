"""Exercise the running ensemble API and verify its established JSON contract."""

import base64
import json
import os
import random
import time
import urllib.request
import uuid
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


API_URL = os.getenv(
    "API_URL", "http://127.0.0.1:8005/api/v1/predict"
)
IMAGE_DIR = Path("/test_images")
SAMPLE_SIZE = int(os.getenv("SMOKE_SAMPLE_SIZE", "20"))
SAMPLE_SEED = 20260724
TOP_LEVEL_KEYS = {"filename", "predictions", "annotated_image"}
PREDICTION_KEYS = {
    "predicted_class",
    "predicted_grade",
    "confidence",
    "description",
    "details",
    "box",
    "yolo_confidence",
    "knee_side",
    "roi_image",
    "gradcam_image",
}
DETAIL_KEYS = {"0Normal", "1Doubtful", "2Mild", "3Moderate", "4Severe"}


def multipart_body(image_path: Path) -> tuple[bytes, str]:
    boundary = "----CodexBoundary" + uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{image_path.name}"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode("ascii")
    body += image_path.read_bytes()
    body += f"\r\n--{boundary}--\r\n".encode("ascii")
    return body, f"multipart/form-data; boundary={boundary}"


def decode_data_image(value: str, expected_size: tuple[int, int] | None = None) -> None:
    if not value.startswith("data:image/") or "," not in value:
        raise AssertionError("Expected an image data URL")
    raw = base64.b64decode(value.split(",", 1)[1], validate=True)
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise AssertionError("Image data URL could not be decoded")
    if expected_size and image.shape[:2] != expected_size:
        raise AssertionError(
            f"Expected image size {expected_size}, received {image.shape[:2]}"
        )


def main() -> None:
    image_paths = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    selected = (
        image_paths
        if SAMPLE_SIZE <= 0 or SAMPLE_SIZE >= len(image_paths)
        else sorted(random.Random(SAMPLE_SEED).sample(image_paths, SAMPLE_SIZE))
    )

    rows = []
    all_grades = []
    total_predictions = 0
    for image_path in selected:
        body, content_type = multipart_body(image_path)
        request = urllib.request.Request(
            API_URL,
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=180) as response:
            status = response.status
            payload = json.loads(response.read())
        elapsed = time.perf_counter() - started

        if status != 200:
            raise AssertionError(f"Unexpected HTTP status {status}")
        if set(payload) != TOP_LEVEL_KEYS:
            raise AssertionError(f"Top-level schema changed: {set(payload)}")
        if payload["filename"] != image_path.name:
            raise AssertionError("Response filename does not match request")
        if not payload["predictions"]:
            raise AssertionError("No knee predictions returned")
        decode_data_image(payload["annotated_image"])

        grades = []
        confidences = []
        for prediction in payload["predictions"]:
            if set(prediction) != PREDICTION_KEYS:
                raise AssertionError(
                    f"Prediction schema changed: {set(prediction)}"
                )
            if set(prediction["details"]) != DETAIL_KEYS:
                raise AssertionError("Class-probability detail keys changed")
            if not 0 <= prediction["predicted_class"] <= 4:
                raise AssertionError("Predicted class is outside KL grades 0-4")
            probability_sum = sum(prediction["details"].values())
            if abs(probability_sum - 1.0) > 1e-5:
                raise AssertionError(f"Probabilities sum to {probability_sum}")
            decode_data_image(prediction["gradcam_image"], expected_size=(384, 384))
            if prediction["roi_image"] is not None:
                decode_data_image(prediction["roi_image"])
            grades.append(prediction["predicted_class"])
            confidences.append(prediction["confidence"])

        total_predictions += len(grades)
        all_grades.extend(grades)
        rows.append(
            {
                "image": image_path.name,
                "status": status,
                "knees": len(grades),
                "grades": grades,
                "confidences": confidences,
                "seconds": elapsed,
            }
        )

    result = {
        "sample_seed": SAMPLE_SEED,
        "images": len(selected),
        "predictions": total_predictions,
        "schema_unchanged": True,
        "all_heatmaps_decoded_at_384x384": True,
        "grade_distribution": dict(sorted(Counter(all_grades).items())),
        "mean_request_seconds": sum(row["seconds"] for row in rows) / len(rows),
        "max_request_seconds": max(row["seconds"] for row in rows),
        "rows": rows,
    }
    print("SMOKE_JSON=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
