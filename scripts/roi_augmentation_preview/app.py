from __future__ import annotations

import math
import secrets
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response


APP_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
OUTPUT_SIZE = 384

app = FastAPI(
    title="Knee ROI Augmentation Preview",
    version="1.0.0",
)


def apply_clahe_1_25(image_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.25, tileGridSize=(8, 8))
    enhanced = clahe.apply(lightness)
    return cv2.cvtColor(
        cv2.merge((enhanced, channel_a, channel_b)),
        cv2.COLOR_LAB2RGB,
    )


def square_pad(image_rgb: np.ndarray) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    side = max(height, width)
    top = (side - height) // 2
    bottom = side - height - top
    left = (side - width) // 2
    right = side - width - left
    return cv2.copyMakeBorder(
        image_rgb,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def adjust_brightness(image_rgb: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(image_rgb.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def adjust_contrast(image_rgb: np.ndarray, factor: float) -> np.ndarray:
    grayscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mean = float(grayscale.mean())
    adjusted = image_rgb.astype(np.float32) * factor + mean * (1.0 - factor)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def random_erasing(
    image_rgb: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str | None]:
    if rng.random() >= 0.10:
        return image_rgb, None

    height, width = image_rgb.shape[:2]
    area = height * width
    for _ in range(10):
        target_area = area * rng.uniform(0.02, 0.05)
        aspect_ratio = math.exp(rng.uniform(math.log(0.5), math.log(2.0)))
        erase_height = int(round(math.sqrt(target_area * aspect_ratio)))
        erase_width = int(round(math.sqrt(target_area / aspect_ratio)))
        if 0 < erase_height < height and 0 < erase_width < width:
            top = int(rng.integers(0, height - erase_height + 1))
            left = int(rng.integers(0, width - erase_width + 1))
            result = image_rgb.copy()
            result[top : top + erase_height, left : left + erase_width] = 0
            return result, f"erase({left},{top},{erase_width},{erase_height})"
    return image_rgb, None


def augment_roi(
    image_rgb: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    """Mirror the current DenseNet training augmentation before normalization."""
    rng = np.random.default_rng(seed)
    operations = ["clahe=1.25", "square_pad"]

    result = square_pad(apply_clahe_1_25(image_rgb))
    if rng.random() < 0.50:
        result = np.ascontiguousarray(result[:, ::-1])
        operations.append("horizontal_flip")

    angle = float(rng.uniform(-5.0, 5.0))
    center = ((result.shape[1] - 1) / 2.0, (result.shape[0] - 1) / 2.0)
    rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(
        result,
        rotation,
        (result.shape[1], result.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    operations.append(f"rotation={angle:.3f}")

    brightness = float(rng.uniform(0.92, 1.08))
    contrast = float(rng.uniform(0.92, 1.08))
    intensity_operations = [
        ("brightness", brightness, adjust_brightness),
        ("contrast", contrast, adjust_contrast),
    ]
    rng.shuffle(intensity_operations)
    for name, factor, operation in intensity_operations:
        result = operation(result, factor)
        operations.append(f"{name}={factor:.3f}")

    result = cv2.resize(
        result,
        (OUTPUT_SIZE, OUTPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    result, erase_description = random_erasing(result, rng)
    if erase_description:
        operations.append(erase_description)
    operations.append(f"resize={OUTPUT_SIZE}")
    return result, operations


def decode_image(payload: bytes) -> np.ndarray:
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 20 MB limit.")
    image_bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="File is not a decodable image.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(APP_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "healthy",
        "output_size": OUTPUT_SIZE,
        "pipeline": "CLAHE 1.25 -> SquarePad -> training augmentation -> Resize 384",
    }


@app.post(
    "/augment",
    responses={200: {"content": {"image/png": {}}}},
    response_class=Response,
)
async def augment(
    image: UploadFile = File(...),
    seed: int | None = Query(default=None, ge=0, le=2**32 - 1),
) -> Response:
    payload = await image.read(MAX_UPLOAD_BYTES + 1)
    source_rgb = decode_image(payload)
    effective_seed = seed if seed is not None else secrets.randbits(32)
    augmented_rgb, operations = augment_roi(source_rgb, effective_seed)
    success, encoded = cv2.imencode(
        ".png",
        cv2.cvtColor(augmented_rgb, cv2.COLOR_RGB2BGR),
    )
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode augmented image.")
    return Response(
        content=encoded.tobytes(),
        media_type="image/png",
        headers={
            "X-Augmentation-Seed": str(effective_seed),
            "X-Augmentation-Operations": "; ".join(operations),
            "Content-Disposition": f'inline; filename="augmented_{effective_seed}.png"',
        },
    )
