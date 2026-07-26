"""Audit every API prediction and native CAM against the production anatomy gate."""

import base64
import csv
import json
import os
import time
import urllib.request
import uuid
from collections import Counter
from datetime import datetime, timezone
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


API_URL = os.getenv("API_URL", "http://127.0.0.1:8005/api/v1/predict")
IMAGE_DIR = Path(os.getenv("IMAGE_DIR", "/test_images"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/tmp/api_cam_localization_audit"))
RUN_TIMESTAMP = os.getenv(
    "RUN_TIMESTAMP", datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_UTC")
)
ASSET_PREFIX = os.getenv("REPORT_ASSET_PREFIX", "assets/api_cam_localization_audit")
ROWS_PER_MONTAGE = int(os.getenv("ROWS_PER_MONTAGE", "10"))

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


def multipart_body(image_path: Path) -> tuple[bytes, str]:
    boundary = "----CamAuditBoundary" + uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{image_path.name}"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode("ascii")
    body += image_path.read_bytes()
    body += f"\r\n--{boundary}--\r\n".encode("ascii")
    return body, f"multipart/form-data; boundary={boundary}"


def call_api(image_path: Path) -> tuple[dict, float]:
    body, content_type = multipart_body(image_path)
    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        if response.status != 200:
            raise RuntimeError(f"{image_path.name}: HTTP {response.status}")
        payload = json.loads(response.read())
    elapsed = time.perf_counter() - started
    if set(payload) != TOP_LEVEL_KEYS:
        raise RuntimeError(f"{image_path.name}: top-level response schema changed")
    if payload["filename"] != image_path.name:
        raise RuntimeError(f"{image_path.name}: response filename mismatch")
    if not payload["predictions"]:
        raise RuntimeError(f"{image_path.name}: API returned no knee predictions")
    for prediction in payload["predictions"]:
        if set(prediction) != PREDICTION_KEYS:
            raise RuntimeError(f"{image_path.name}: prediction schema changed")
    return payload, elapsed


def decode_data_image(value: str) -> np.ndarray:
    if not value.startswith("data:image/") or "," not in value:
        raise ValueError("Expected an image data URL")
    raw = base64.b64decode(value.split(",", 1)[1], validate=True)
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode API image")
    return image


def sides_for_knees(knees: list[dict], image_width: int) -> tuple[list[dict], list[str]]:
    if len(knees) == 2:
        knees = sorted(knees, key=lambda knee: knee["box"][0])
        return knees, ["right", "left"]
    if len(knees) == 1:
        x1, _, x2, _ = knees[0]["box"]
        center_x = (x1 + x2) / 2
        if center_x < image_width * 0.40:
            return knees, ["right"]
        if center_x > image_width * 0.60:
            return knees, ["left"]
        return knees, ["unknown"]
    return knees, ["unknown"] * len(knees)


def diagnostic_predictions(image_path: Path) -> list[dict]:
    image_bytes = image_path.read_bytes()
    original = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError(f"Could not decode {image_path}")
    knees = roi_service.detect_knees_with_coords(image_bytes)
    if knees:
        knees, sides = sides_for_knees(knees, original.shape[1])
    else:
        knees = [{"box": None, "crop_bytes": image_bytes, "yolo_conf": 0.0}]
        sides = ["unknown"]

    pipeline = prediction_service.pipeline
    if len(pipeline.models) != 1 or "densenet121" not in pipeline.models:
        raise RuntimeError(
            f"Audit requires DenseNet-only mode; loaded {list(pipeline.models)}"
        )
    model = pipeline.models["densenet121"]
    results = []
    for knee, side in zip(knees, sides):
        tensor, processed_image, _ = preprocessing_service.preprocess_image(
            knee["crop_bytes"], knee_side=side
        )
        tensor = tensor.to(pipeline.device)
        logits, class_maps = inference_service.run_inference_with_class_maps(
            model, tensor
        )
        probabilities = F.softmax(logits.float(), dim=1)
        predicted_class = int(probabilities.argmax(dim=1).item())
        cam = native_cam_service.extract_cam(
            model=model,
            class_maps=class_maps,
            predicted_class=predicted_class,
            output_size=processed_image.shape[:2],
        )
        metrics = native_cam_service.energy_metrics(cam)
        box = knee["box"]
        if box is None:
            roi_width, roi_height, padding_fraction = original.shape[1], original.shape[0], 0.0
        else:
            roi_width = int(box[2] - box[0])
            roi_height = int(box[3] - box[1])
            square_side = max(roi_width, roi_height)
            padding_fraction = 1.0 - (roi_width * roi_height) / (square_side**2)
        results.append(
            {
                "predicted_class": predicted_class,
                "confidence": float(probabilities[0, predicted_class].item()),
                "box": box,
                "yolo_confidence": float(knee["yolo_conf"]),
                "knee_side": side,
                "processed_image": processed_image,
                "cam": cam,
                "metrics": metrics,
                "roi_width": roi_width,
                "roi_height": roi_height,
                "roi_aspect_width_over_height": roi_width / max(roi_height, 1),
                "square_padding_fraction": padding_fraction,
            }
        )
    return results


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


def peak_patterns(metrics: dict) -> list[str]:
    patterns = []
    if metrics["peak_y"] < 0.28:
        patterns.append("upper_femur_or_top_border")
    elif metrics["peak_y"] >= 0.72:
        patterns.append("lower_tibia_or_bottom_border")
    if metrics["peak_x"] < 0.06 or metrics["peak_x"] >= 0.94:
        patterns.append("far_lateral_border")
    if not patterns and metrics["joint_energy"] < MIN_HEATMAP_JOINT_ENERGY:
        patterns.append("diffuse_or_off_joint")
    return patterns or ["threshold_only"]


def safe_stem(filename: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in filename)


def render_failure_tile(row: dict, roi_image: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    roi_image = cv2.resize(roi_image, (384, 384), interpolation=cv2.INTER_AREA)
    overlay = cv2.resize(overlay, (384, 384), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((450, 768, 3), dtype=np.uint8)
    canvas[66:, :384] = roi_image
    canvas[66:, 384:] = overlay
    title = (
        f"{row['filename']} | {row['knee_side']} | G{row['predicted_grade']} "
        f"p={row['confidence']:.3f}"
    )
    metrics = (
        f"joint={row['joint_energy']:.3f} border={row['border_energy']:.3f} "
        f"lower={row['lower_tibia_energy']:.3f} peak=({row['peak_x']:.2f},"
        f"{row['peak_y']:.2f})"
    )
    cv2.putText(canvas, title[:105], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, metrics, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "ROI", (8, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "API native-CAM overlay", (392, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def mean(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows])) if rows else float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| # | Image | Side | Pred. | Conf. | Joint | Border | Lower tibia | Peak x,y | Failure reasons |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"| {index} | `{row['filename']}` | {row['knee_side']} | "
            f"G{row['predicted_grade']} | {row['confidence']:.3f} | "
            f"{row['joint_energy']:.3f} | {row['border_energy']:.3f} | "
            f"{row['lower_tibia_energy']:.3f} | "
            f"{row['peak_x']:.2f}, {row['peak_y']:.2f} | "
            f"{row['failure_reasons'].replace('_', ' ')} |"
        )
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    failed_case_dir = OUTPUT_DIR / "failed_cases"
    failed_case_dir.mkdir()
    image_paths = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not image_paths:
        raise RuntimeError(f"No test images found in {IMAGE_DIR}")

    all_rows = []
    failed_rows = []
    failure_tiles = []
    request_seconds = []
    for image_index, image_path in enumerate(image_paths, start=1):
        payload, elapsed = call_api(image_path)
        request_seconds.append(elapsed)
        diagnostics = diagnostic_predictions(image_path)
        if len(payload["predictions"]) != len(diagnostics):
            raise RuntimeError(
                f"{image_path.name}: API/direct knee count mismatch "
                f"{len(payload['predictions'])} != {len(diagnostics)}"
            )
        for knee_index, (api_prediction, diagnostic) in enumerate(
            zip(payload["predictions"], diagnostics), start=1
        ):
            if api_prediction["predicted_class"] != diagnostic["predicted_class"]:
                raise RuntimeError(f"{image_path.name}: API/direct grade mismatch")
            if api_prediction["knee_side"] != diagnostic["knee_side"]:
                raise RuntimeError(f"{image_path.name}: API/direct side mismatch")
            if abs(api_prediction["confidence"] - diagnostic["confidence"]) > 1e-5:
                raise RuntimeError(f"{image_path.name}: API/direct confidence mismatch")

            metrics = diagnostic["metrics"]
            reasons = failure_reasons(metrics)
            patterns = peak_patterns(metrics)
            row = {
                "filename": image_path.name,
                "knee_index": knee_index,
                "knee_side": diagnostic["knee_side"],
                "predicted_grade": diagnostic["predicted_class"],
                "confidence": diagnostic["confidence"],
                "yolo_confidence": diagnostic["yolo_confidence"],
                "box": json.dumps(diagnostic["box"]),
                "roi_width": diagnostic["roi_width"],
                "roi_height": diagnostic["roi_height"],
                "roi_aspect_width_over_height": diagnostic[
                    "roi_aspect_width_over_height"
                ],
                "square_padding_fraction": diagnostic["square_padding_fraction"],
                "joint_energy": metrics["joint_energy"],
                "joint_enrichment": metrics["joint_enrichment"],
                "border_energy": metrics["border_energy"],
                "border_enrichment": metrics["border_enrichment"],
                "lower_tibia_energy": metrics["lower_tibia_energy"],
                "peak_x": metrics["peak_x"],
                "peak_y": metrics["peak_y"],
                "peak_inside_joint": metrics["peak_inside_joint"],
                "anatomy_score": metrics["anatomy_score"],
                "gate_pass": not reasons,
                "failure_reasons": ";".join(reasons),
                "peak_patterns": ";".join(patterns),
                "failure_image": "",
                "api_schema_unchanged": True,
            }
            all_rows.append(row)
            if reasons:
                roi_image = (
                    decode_data_image(api_prediction["roi_image"])
                    if api_prediction["roi_image"] is not None
                    else cv2.cvtColor(
                        diagnostic["processed_image"], cv2.COLOR_RGB2BGR
                    )
                )
                overlay = decode_data_image(api_prediction["gradcam_image"])
                file_stem = (
                    f"{len(failed_rows) + 1:03d}_{safe_stem(image_path.stem)}_"
                    f"{diagnostic['knee_side']}"
                )
                failure_path = failed_case_dir / f"{file_stem}.jpg"
                tile = render_failure_tile(row, roi_image, overlay)
                if not cv2.imwrite(str(failure_path), tile):
                    raise RuntimeError(f"Could not save {failure_path}")
                row["failure_image"] = f"failed_cases/{failure_path.name}"
                failed_rows.append(row)
                failure_tiles.append(tile)
        print(
            f"[{image_index:03d}/{len(image_paths)}] {image_path.name}: "
            f"{len(payload['predictions'])} knee(s)"
        )

    write_csv(OUTPUT_DIR / "all_knees_cam_audit.csv", all_rows)
    write_csv(OUTPUT_DIR / "failed_cams.csv", failed_rows)

    montage_paths = []
    for start in range(0, len(failure_tiles), ROWS_PER_MONTAGE):
        page = np.vstack(failure_tiles[start : start + ROWS_PER_MONTAGE])
        page_number = len(montage_paths) + 1
        path = OUTPUT_DIR / f"failed_cam_montage_{page_number:02d}.jpg"
        if not cv2.imwrite(str(path), page, [cv2.IMWRITE_JPEG_QUALITY, 91]):
            raise RuntimeError(f"Could not save {path}")
        montage_paths.append(path)

    reason_counts = Counter(
        reason for row in failed_rows for reason in row["failure_reasons"].split(";")
    )
    pattern_counts = Counter(
        pattern for row in failed_rows for pattern in row["peak_patterns"].split(";")
    )
    grade_counts = Counter(row["predicted_grade"] for row in all_rows)
    failed_grade_counts = Counter(row["predicted_grade"] for row in failed_rows)
    unique_failed_images = len({row["filename"] for row in failed_rows})
    passed_rows = [row for row in all_rows if row["gate_pass"]]

    report_lines = [
        "# DenseNet-121 Full API Native-CAM Localization Audit",
        "",
        f"Exact audit timestamp: `{RUN_TIMESTAMP}`.",
        "",
        "## Scope and Method",
        "",
        f"All `{len(image_paths)}` files in `test_images` were submitted to the live "
        f"`/api/v1/predict` endpoint. The endpoint returned `{len(all_rows)}` knee "
        "predictions. Each response was checked against the established JSON schema, "
        "and every API prediction was cross-checked against the same mounted "
        "DenseNet/YOLO pipeline to obtain the raw native-CAM metrics.",
        "",
        "The localization gate passes only when joint energy is at least `0.55`, "
        "border energy is at most `0.25`, lower-tibia energy is at most `0.25`, "
        "and the maximum CAM activation lies inside the broad joint band. This is "
        "an engineering anatomy heuristic, not an expert lesion annotation.",
        "",
        "The test images have no KL ground-truth labels. This audit measures API "
        "operation and CAM geometry; it does not measure classification accuracy.",
        "",
        "## Summary",
        "",
        "| Item | Result |",
        "| --- | ---: |",
        f"| API images completed | {len(image_paths)} / {len(image_paths)} |",
        f"| Knee predictions | {len(all_rows)} |",
        f"| CAMs passing gate | {len(passed_rows)} / {len(all_rows)} ({len(passed_rows) / len(all_rows):.1%}) |",
        f"| CAMs failing gate | {len(failed_rows)} / {len(all_rows)} ({len(failed_rows) / len(all_rows):.1%}) |",
        f"| Unique images with at least one failed CAM | {unique_failed_images} / {len(image_paths)} |",
        f"| Mean API request time | {np.mean(request_seconds):.3f} seconds |",
        f"| Maximum API request time | {np.max(request_seconds):.3f} seconds |",
        "| API schema | Unchanged |",
        "",
        "## Common Failure Criteria",
        "",
        "Counts overlap because one CAM can fail multiple criteria.",
        "",
        "| Criterion | Failed CAMs |",
        "| --- | ---: |",
    ]
    for reason, count in reason_counts.most_common():
        report_lines.append(f"| {reason.replace('_', ' ')} | {count} / {len(failed_rows)} |")
    report_lines.extend(
        [
            "",
            "## Common Spatial Patterns",
            "",
            "| Pattern | Failed CAMs |",
            "| --- | ---: |",
        ]
    )
    for pattern, count in pattern_counts.most_common():
        report_lines.append(f"| {pattern.replace('_', ' ')} | {count} / {len(failed_rows)} |")
    report_lines.extend(
        [
            "",
            "## Aggregate Geometry",
            "",
            "| Group | Joint energy | Border energy | Lower-tibia energy | Anatomy score | Square-padding fraction |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            f"| Passing CAMs | {mean(passed_rows, 'joint_energy'):.3f} | {mean(passed_rows, 'border_energy'):.3f} | {mean(passed_rows, 'lower_tibia_energy'):.3f} | {mean(passed_rows, 'anatomy_score'):.3f} | {mean(passed_rows, 'square_padding_fraction'):.3f} |",
            f"| Failed CAMs | {mean(failed_rows, 'joint_energy'):.3f} | {mean(failed_rows, 'border_energy'):.3f} | {mean(failed_rows, 'lower_tibia_energy'):.3f} | {mean(failed_rows, 'anatomy_score'):.3f} | {mean(failed_rows, 'square_padding_fraction'):.3f} |",
            "",
            "## Prediction Distribution",
            "",
            "| Predicted grade | All CAMs | Failed CAMs | Failure rate |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for grade in range(5):
        total = grade_counts[grade]
        failed = failed_grade_counts[grade]
        rate = failed / total if total else 0.0
        report_lines.append(f"| {grade} | {total} | {failed} | {rate:.1%} |")
    report_lines.extend(
        [
            "",
            "## Visual Review Pages",
            "",
        ]
    )
    for index, montage_path in enumerate(montage_paths, start=1):
        report_lines.extend(
            [
                f"### Failed CAM Montage {index}",
                "",
                f"![Failed CAM montage {index}]({ASSET_PREFIX}/{montage_path.name})",
                "",
            ]
        )
    report_lines.extend(
        [
            "## Every Failed Knee CAM",
            "",
            "The complete machine-readable files are "
            f"[`failed_cams.csv`]({ASSET_PREFIX}/failed_cams.csv) and "
            f"[`all_knees_cam_audit.csv`]({ASSET_PREFIX}/all_knees_cam_audit.csv).",
            "",
            markdown_table(failed_rows),
            "",
            "## Interpretation",
            "",
            "A failed gate result does not prove that the KL prediction is incorrect, "
            "and a passing result does not prove lesion-level localization. The common "
            "failure modes identify where the selected-class evidence is geometrically "
            "inconsistent with the tibiofemoral joint. The raw images are unlabeled, "
            "so clinical correctness requires expert review or a labeled external set.",
            "",
            "### Common mistakes observed",
            "",
            f"1. **Upper-femur and top-edge shortcut:** `{pattern_counts['upper_femur_or_top_border']} / {len(failed_rows)}` failed CAMs peak on the superior femur or top image boundary.",
            f"2. **Lateral crop-edge shortcut:** `{pattern_counts['far_lateral_border']} / {len(failed_rows)}` peak at the far left or right boundary.",
            f"3. **Lower-tibia shortcut:** `{pattern_counts['lower_tibia_or_bottom_border']} / {len(failed_rows)}` emphasize the inferior tibia or bottom boundary.",
            f"4. **Diffuse evidence:** `{pattern_counts['diffuse_or_off_joint']} / {len(failed_rows)}` spread weak activation across multiple off-joint regions. Per-image min-max color normalization can make a weak maximum appear strongly red.",
            f"5. **Grade 0 is especially problematic:** `{failed_grade_counts[0]} / {grade_counts[0]}` Grade 0 CAMs fail. Absence-of-disease evidence is not guaranteed to form a lesion map.",
            "6. **No visual correction should be substituted for evidence:** clipping the CAM to a joint mask would conceal off-joint evidence rather than correct the classifier.",
            "",
            "The final feature grid is coarse and becomes blocky when enlarged to the ROI. This explains limited spatial resolution, but repeated off-joint peaks indicate learned non-joint evidence from the training distribution.",
            "",
            "### Recommended next action",
            "",
            "Retrain using the exact production YOLO crop pipeline, limited ROI scale/translation augmentation, and explicit localization supervision or an auxiliary joint-mask loss. Compare with the current checkpoint on a labeled, patient-separated holdout using both KL metrics and these frozen CAM criteria. Until then, describe the output as model attention rather than lesion localization.",
            "",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    summary = {
        "run_timestamp": RUN_TIMESTAMP,
        "images": len(image_paths),
        "knees": len(all_rows),
        "passed_cams": len(passed_rows),
        "failed_cams": len(failed_rows),
        "unique_failed_images": unique_failed_images,
        "reason_counts": dict(reason_counts),
        "pattern_counts": dict(pattern_counts),
        "grade_counts": dict(grade_counts),
        "failed_grade_counts": dict(failed_grade_counts),
        "mean_request_seconds": float(np.mean(request_seconds)),
        "max_request_seconds": float(np.max(request_seconds)),
        "api_schema_unchanged": True,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("AUDIT_SUMMARY=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
