"""Export failed production DenseNet native-CAM cases with raw gate metrics."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch.nn.functional as F

from app.ml.pipelines.knee_oa_pipeline import (
    MAX_HEATMAP_BORDER_ENERGY,
    MAX_HEATMAP_LOWER_TIBIA_ENERGY,
    MIN_HEATMAP_JOINT_ENERGY,
)
from app.services.gradcam_service import native_cam_service
from app.services.inference_service import inference_service
from app.services.prediction_service import prediction_service
from app.services.preprocessing_service import preprocessing_service
from app.services.roi_service import roi_service


IMAGE_DIR = Path(os.getenv("IMAGE_DIR", "/tmp/test_images"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/tmp/densenet_cam_audit"))
ROWS_PER_MONTAGE = int(os.getenv("ROWS_PER_MONTAGE", "8"))


def knee_sides(knees: list[dict], image_width: int) -> tuple[list[dict], list[str]]:
    if len(knees) == 2:
        return sorted(knees, key=lambda knee: knee["box"][0]), ["right", "left"]
    if len(knees) == 1:
        x1, _, x2, _ = knees[0]["box"]
        center_x = (x1 + x2) / 2
        if center_x < image_width * 0.40:
            return knees, ["right"]
        if center_x > image_width * 0.60:
            return knees, ["left"]
        return knees, ["unknown"]
    return knees, ["unknown"] * len(knees)


def failure_reasons(metrics: dict) -> list[str]:
    reasons = []
    if metrics["joint_energy"] < MIN_HEATMAP_JOINT_ENERGY:
        reasons.append("low_joint_energy")
    if metrics["border_energy"] > MAX_HEATMAP_BORDER_ENERGY:
        reasons.append("high_border_energy")
    if metrics["lower_tibia_energy"] > MAX_HEATMAP_LOWER_TIBIA_ENERGY:
        reasons.append("high_lower_tibia_energy")
    if not metrics["peak_inside_joint"]:
        reasons.append("peak_outside_joint")
    return reasons


def fit_square(image_bgr: np.ndarray, size: int = 384) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    scale = min(size / width, size / height)
    resized = cv2.resize(
        image_bgr,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def render_tile(
    row: dict,
    roi_bgr: np.ndarray,
    processed_rgb: np.ndarray,
    cam: np.ndarray,
) -> np.ndarray:
    size, header = 384, 82
    roi = fit_square(roi_bgr, size)
    processed = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
    heat = cv2.applyColorMap(np.uint8(np.clip(cam, 0, 1) * 255), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(processed, 0.60, heat, 0.40, 0)
    tile = np.zeros((header + size, size * 3, 3), dtype=np.uint8)
    tile[header:, :size] = roi
    tile[header:, size : size * 2] = processed
    tile[header:, size * 2 :] = overlay
    title = (
        f"{row['filename']} | {row['knee_side']} | G{row['predicted_grade']} "
        f"p={row['confidence']:.3f} | YOLO={row['yolo_confidence']:.3f}"
    )
    metrics = (
        f"joint={row['joint_energy']:.3f} border={row['border_energy']:.3f} "
        f"lower={row['lower_tibia_energy']:.3f} peak=({row['peak_x']:.2f},"
        f"{row['peak_y']:.2f}) | {row['failure_reasons']}"
    )
    cv2.putText(tile, title[:130], (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(tile, metrics[:155], (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    for x, label in ((8, "YOLO ROI"), (size + 8, "Exact model input"), (size * 2 + 8, "Native CAM")):
        cv2.putText(tile, label, (x, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def safe_stem(filename: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in filename)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    failure_dir = OUTPUT_DIR / "failed_heatmaps"
    failure_dir.mkdir()
    image_paths = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {IMAGE_DIR}")

    pipeline = prediction_service.pipeline
    if list(pipeline.models) != ["densenet121"]:
        raise RuntimeError(f"DenseNet-only audit required; loaded {list(pipeline.models)}")
    model = pipeline.models["densenet121"]
    rows, failed_rows, failed_tiles = [], [], []

    for image_index, image_path in enumerate(image_paths, start=1):
        image_bytes = image_path.read_bytes()
        source = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if source is None:
            raise ValueError(f"Could not decode {image_path}")
        knees = roi_service.detect_knees_with_coords(image_bytes)
        knees, sides = knee_sides(knees, source.shape[1])
        for knee_index, (knee, side) in enumerate(zip(knees, sides)):
            tensor, processed, _ = preprocessing_service.preprocess_image(
                knee["crop_bytes"], knee_side=side
            )
            tensor = tensor.to(pipeline.device)
            logits, class_maps = inference_service.run_inference_with_class_maps(
                model, tensor
            )
            probabilities = F.softmax(logits.float(), dim=1)
            predicted_grade = int(probabilities.argmax(1).item())
            cam = native_cam_service.extract_cam(
                model=model,
                class_maps=class_maps,
                predicted_class=predicted_grade,
                output_size=processed.shape[:2],
            )
            metrics = native_cam_service.energy_metrics(cam)
            reasons = failure_reasons(metrics)
            row = {
                "filename": image_path.name,
                "knee_index": knee_index,
                "knee_side": side,
                "predicted_grade": predicted_grade,
                "confidence": float(probabilities[0, predicted_grade].item()),
                "yolo_confidence": float(knee["yolo_conf"]),
                "box": json.dumps(knee["box"]),
                "gate_pass": int(not reasons),
                "failure_reasons": ";".join(reasons),
                "exported_heatmap": "",
                **metrics,
            }
            rows.append(row)
            if reasons:
                roi = cv2.imdecode(
                    np.frombuffer(knee["crop_bytes"], np.uint8), cv2.IMREAD_COLOR
                )
                tile = render_tile(row, roi, processed, cam)
                output_name = (
                    f"{safe_stem(image_path.stem)}__{side}__G{predicted_grade}"
                    f"__failure_{len(failed_rows) + 1:03d}.jpg"
                )
                cv2.imwrite(str(failure_dir / output_name), tile)
                row["exported_heatmap"] = f"failed_heatmaps/{output_name}"
                failed_rows.append(row.copy())
                failed_tiles.append(tile)
        print(f"[{image_index:03d}/{len(image_paths)}] knees={len(knees)}", flush=True)

    write_csv(OUTPUT_DIR / "all_cases.csv", rows)
    write_csv(OUTPUT_DIR / "failed_cases.csv", failed_rows)
    for page, start in enumerate(range(0, len(failed_tiles), ROWS_PER_MONTAGE), 1):
        montage = np.vstack(failed_tiles[start : start + ROWS_PER_MONTAGE])
        cv2.imwrite(str(OUTPUT_DIR / f"failed_heatmaps_montage_{page:02d}.jpg"), montage)

    summary = {
        "images": len(image_paths),
        "predictions": len(rows),
        "gate_passes": sum(row["gate_pass"] for row in rows),
        "gate_failures": len(failed_rows),
        "gate_pass_rate": float(np.mean([row["gate_pass"] for row in rows])),
        "grade_distribution": dict(Counter(row["predicted_grade"] for row in rows)),
        "failed_grade_distribution": dict(
            Counter(row["predicted_grade"] for row in failed_rows)
        ),
        "failure_reasons": dict(
            Counter(reason for row in failed_rows for reason in row["failure_reasons"].split(";") if reason)
        ),
        "checkpoint": pipeline.checkpoint_metadata,
        "preprocessing": "CLAHE 1.25 -> SquarePad -> Resize 384",
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
