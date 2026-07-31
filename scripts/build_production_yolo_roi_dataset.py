#!/usr/bin/env python3
"""Build a labeled knee ROI dataset with the production YOLO crop contract.

The full radiograph filename must begin with the OAI patient ID. Labels are
resolved from Kaggle knee crop names of the form ``<patient><R|L>.png``. The
legacy ``<patient>_<1|2>.png`` convention is also accepted, where 1 is the
anatomical right knee and 2 is the anatomical left knee.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
KAGGLE_NAME = re.compile(
    r"^(?P<patient>\d+)(?:(?P<letter>[RL])|_(?P<number>[12]))$",
    re.IGNORECASE,
)
FULL_IMAGE_PATIENT = re.compile(r"^(?P<patient>\d+)(?:_|$)")
SIDE_NUMBER = {"right": "1", "left": "2"}
CSV_FIELDS = [
    "patient_id",
    "knee_side",
    "kaggle_side_number",
    "split",
    "grade",
    "source_full_image",
    "source_kaggle_crop",
    "output_roi",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "roi_width",
    "roi_height",
    "roi_aspect_ratio",
    "yolo_confidence",
    "source_sha256",
    "roi_sha256",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-images", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--yolo-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument("--montage-count", type=int, default=40)
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def index_kaggle_labels(labels_root: Path) -> tuple[dict, list[dict]]:
    labels: dict[tuple[str, str], dict] = {}
    conflicts: list[dict] = []
    for split in ("train", "val", "test"):
        split_dir = labels_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing Kaggle split: {split_dir}")
        for grade_dir in sorted(split_dir.iterdir()):
            if not grade_dir.is_dir() or grade_dir.name not in {"0", "1", "2", "3", "4"}:
                continue
            for path in image_paths(grade_dir):
                match = KAGGLE_NAME.match(path.stem)
                if not match:
                    continue
                side_number = match.group("number") or (
                    "1" if match.group("letter").upper() == "R" else "2"
                )
                key = (match.group("patient"), side_number)
                candidate = {
                    "split": split,
                    "grade": int(grade_dir.name),
                    "path": path,
                }
                previous = labels.get(key)
                if previous and (
                    previous["split"] != split or previous["grade"] != candidate["grade"]
                ):
                    conflicts.append(
                        {
                            "patient_id": key[0],
                            "kaggle_side_number": key[1],
                            "reason": "conflicting_kaggle_label_or_split",
                            "details": f"{previous} versus {candidate}",
                        }
                    )
                    labels.pop(key, None)
                elif key not in labels:
                    labels[key] = candidate

    patient_splits: dict[str, set[str]] = defaultdict(set)
    for (patient_id, _), value in labels.items():
        patient_splits[patient_id].add(value["split"])
    conflicting_patients = {
        patient_id for patient_id, splits in patient_splits.items() if len(splits) > 1
    }
    if conflicting_patients:
        for key in list(labels):
            if key[0] in conflicting_patients:
                conflicts.append(
                    {
                        "patient_id": key[0],
                        "kaggle_side_number": key[1],
                        "reason": "patient_crosses_kaggle_splits",
                        "details": ";".join(sorted(patient_splits[key[0]])),
                    }
                )
                labels.pop(key)
    return labels, conflicts


def assign_sides(boxes: list[dict], image_width: int) -> list[dict]:
    boxes = sorted(boxes, key=lambda item: item["box"][0])
    if len(boxes) == 2:
        boxes[0]["side"] = "right"
        boxes[1]["side"] = "left"
        return boxes
    if len(boxes) == 1:
        x1, _, x2, _ = boxes[0]["box"]
        center_x = (x1 + x2) / 2
        if center_x < image_width * 0.40:
            boxes[0]["side"] = "right"
        elif center_x > image_width * 0.60:
            boxes[0]["side"] = "left"
        else:
            boxes[0]["side"] = "unknown"
        return boxes
    for box in boxes:
        box["side"] = "unknown"
    return boxes


def encode_png(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("OpenCV failed to encode an ROI as PNG")
    return buffer.tobytes()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fit_tile(image: np.ndarray, size: int = 256) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def write_montage(output: Path, samples: list[tuple[np.ndarray, str]]) -> None:
    if not samples:
        return
    tiles = []
    for image, label in samples:
        tile = fit_tile(image)
        cv2.rectangle(tile, (0, 0), (256, 28), (0, 0, 0), -1)
        cv2.putText(
            tile,
            label[:38],
            (6, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    while len(tiles) % 5:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[index : index + 5]) for index in range(0, len(tiles), 5)]
    cv2.imwrite(str(output), np.vstack(rows))


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    labels, rejected = index_kaggle_labels(args.labels_root)
    model = YOLO(str(args.yolo_checkpoint))
    sources = image_paths(args.full_images)
    if not sources:
        raise RuntimeError(f"No full radiographs found in {args.full_images}")

    rows: list[dict] = []
    montage: list[tuple[np.ndarray, str]] = []
    seen_roi_hashes: dict[str, str] = {}
    for source_index, source_path in enumerate(sources, start=1):
        patient_match = FULL_IMAGE_PATIENT.match(source_path.stem)
        if not patient_match:
            rejected.append(
                {
                    "source_full_image": str(source_path),
                    "reason": "patient_id_not_found_in_filename",
                    "details": "",
                }
            )
            continue
        patient_id = patient_match.group("patient")
        source_bytes = source_path.read_bytes()
        source_image = cv2.imdecode(np.frombuffer(source_bytes, np.uint8), cv2.IMREAD_COLOR)
        if source_image is None:
            rejected.append(
                {
                    "patient_id": patient_id,
                    "source_full_image": str(source_path),
                    "reason": "source_decode_failed",
                    "details": "",
                }
            )
            continue

        prediction = model.predict(
            source=source_image,
            conf=args.confidence,
            save=False,
            verbose=False,
        )[0]
        detections = []
        for box in prediction.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append(
                {"box": [x1, y1, x2, y2], "confidence": float(box.conf[0])}
            )
        detections = assign_sides(detections, source_image.shape[1])
        if not detections:
            rejected.append(
                {
                    "patient_id": patient_id,
                    "source_full_image": str(source_path),
                    "reason": "no_yolo_detection",
                    "details": "",
                }
            )

        for detection_index, detection in enumerate(detections):
            side = detection["side"]
            if side == "unknown":
                rejected.append(
                    {
                        "patient_id": patient_id,
                        "source_full_image": str(source_path),
                        "reason": "ambiguous_knee_side",
                        "details": json.dumps(detection["box"]),
                    }
                )
                continue
            side_number = SIDE_NUMBER[side]
            label = labels.get((patient_id, side_number))
            if label is None:
                rejected.append(
                    {
                        "patient_id": patient_id,
                        "knee_side": side,
                        "kaggle_side_number": side_number,
                        "source_full_image": str(source_path),
                        "reason": "matching_kaggle_label_not_found",
                        "details": json.dumps(detection["box"]),
                    }
                )
                continue

            x1, y1, x2, y2 = detection["box"]
            roi = source_image[y1:y2, x1:x2]
            if roi.size == 0:
                rejected.append(
                    {
                        "patient_id": patient_id,
                        "knee_side": side,
                        "source_full_image": str(source_path),
                        "reason": "empty_yolo_crop",
                        "details": json.dumps(detection["box"]),
                    }
                )
                continue
            roi_bytes = encode_png(roi)
            roi_hash = sha256_bytes(roi_bytes)
            duplicate = seen_roi_hashes.get(roi_hash)
            if duplicate:
                rejected.append(
                    {
                        "patient_id": patient_id,
                        "knee_side": side,
                        "source_full_image": str(source_path),
                        "reason": "duplicate_roi_content",
                        "details": duplicate,
                    }
                )
                continue

            relative = Path(label["split"]) / str(label["grade"]) / f"{patient_id}_{side_number}.png"
            output_path = args.output / relative
            if output_path.exists():
                rejected.append(
                    {
                        "patient_id": patient_id,
                        "knee_side": side,
                        "source_full_image": str(source_path),
                        "reason": "duplicate_patient_side",
                        "details": str(output_path),
                    }
                )
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(roi_bytes)
            seen_roi_hashes[roi_hash] = str(relative)
            height, width = roi.shape[:2]
            row = {
                "patient_id": patient_id,
                "knee_side": side,
                "kaggle_side_number": side_number,
                "split": label["split"],
                "grade": label["grade"],
                "source_full_image": str(source_path),
                "source_kaggle_crop": str(label["path"]),
                "output_roi": str(relative),
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "roi_width": width,
                "roi_height": height,
                "roi_aspect_ratio": width / height,
                "yolo_confidence": detection["confidence"],
                "source_sha256": sha256_bytes(source_bytes),
                "roi_sha256": roi_hash,
            }
            rows.append(row)
            if len(montage) < args.montage_count:
                montage.append(
                    (roi, f"{patient_id} {side} | {label['split']} G{label['grade']}")
                )
        print(f"[{source_index:03d}/{len(sources)}] {source_path.name}", flush=True)

    write_csv(args.output / "manifest.csv", rows, CSV_FIELDS)
    rejected_fields = [
        "patient_id",
        "knee_side",
        "kaggle_side_number",
        "source_full_image",
        "reason",
        "details",
    ]
    write_csv(args.output / "rejected.csv", rejected, rejected_fields)
    write_montage(args.output / "roi_montage.jpg", montage)

    split_counts = Counter(row["split"] for row in rows)
    grade_counts = Counter(str(row["grade"]) for row in rows)
    split_grade_counts = Counter(f"{row['split']}/G{row['grade']}" for row in rows)
    rejection_counts = Counter(row["reason"] for row in rejected)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "production-aligned diagnostic ROI dataset",
        "warning": (
            "This subset comes from repeatedly inspected test_images and is not an "
            "independent final test set. Do not report its metrics as generalization."
        ),
        "full_images_root": str(args.full_images),
        "labels_root": str(args.labels_root),
        "yolo_checkpoint": str(args.yolo_checkpoint),
        "yolo_checkpoint_sha256": sha256_file(args.yolo_checkpoint),
        "yolo_confidence_threshold": args.confidence,
        "crop_contract": "exact integer xyxy YOLO crop; no CLAHE, padding, resize, or mirroring baked in",
        "laterality_contract": "two boxes sorted x: leftmost=anatomical right (_1), rightmost=anatomical left (_2)",
        "source_full_images": len(sources),
        "accepted_rois": len(rows),
        "rejected_records": len(rejected),
        "split_counts": dict(sorted(split_counts.items())),
        "grade_counts": dict(sorted(grade_counts.items())),
        "split_grade_counts": dict(sorted(split_grade_counts.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "unique_patients": len({row["patient_id"] for row in rows}),
        "unique_roi_hashes": len(seen_roi_hashes),
        "patient_split_overlap": False,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (args.output / "README.md").write_text(
        "# Production YOLO ROI Diagnostic Dataset\n\n"
        "This dataset was generated from full radiographs using the same YOLO "
        "checkpoint, confidence threshold, integer bounding boxes, and laterality "
        "assignment as the production API. Images are raw detector crops. Apply "
        "the classifier preprocessing at load time.\n\n"
        "> This is a diagnostic/adaptation subset from `test_images`, which has "
        "already been inspected repeatedly. It is not a new locked evaluation set.\n\n"
        "See `manifest.csv`, `rejected.csv`, `metadata.json`, and "
        "`roi_montage.jpg`.\n",
        encoding="utf-8",
    )
    print("SUMMARY=" + json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
