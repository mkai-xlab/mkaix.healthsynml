"""Audit production heatmap selection without changing the API response."""

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from app.ml.pipelines.knee_oa_pipeline import (
    KneeOAPipeline,
    MAX_HEATMAP_BORDER_ENERGY,
    MAX_HEATMAP_LOWER_TIBIA_ENERGY,
    MIN_HEATMAP_JOINT_ENERGY,
    select_heatmap_component,
    weighted_soft_vote,
)
from app.services.gradcam_service import native_cam_service
from app.services.preprocessing_service import preprocessing_service
from app.services.roi_service import roi_service


IMAGE_DIR = Path("/test_images")
KNOWN_FAILURE_IDS = {"9003430", "9063928"}
GLOBAL_LOCALIZATION = {
    "densenet121": 0.7996 * (1.0 - 0.1323),
    "seresnext50_32x4d": 0.8707 * (1.0 - 0.0749),
}


def side_for_roi(knee: dict, width: int, count: int, index: int) -> str:
    if count == 2:
        return ("right", "left")[index]
    center_x = (knee["box"][0] + knee["box"][2]) / 2
    if center_x < width * 0.40:
        return "right"
    if center_x > width * 0.60:
        return "left"
    return "unknown"


def passes(metrics: dict[str, float]) -> bool:
    return bool(
        metrics["joint_energy"] >= MIN_HEATMAP_JOINT_ENERGY
        and metrics["border_energy"] <= MAX_HEATMAP_BORDER_ENERGY
        and metrics["lower_tibia_energy"] <= MAX_HEATMAP_LOWER_TIBIA_ENERGY
        and metrics["peak_inside_joint"]
    )


def old_global_selector(probabilities: dict[str, torch.Tensor], grade: int) -> str:
    agreeing = [
        name
        for name, value in probabilities.items()
        if int(value.argmax(dim=1).item()) == grade
    ]
    candidates = agreeing or list(probabilities)
    return max(
        candidates,
        key=lambda name: (
            float(probabilities[name][0, grade].item())
            * GLOBAL_LOCALIZATION[name]
        ),
    )


def main() -> None:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    pipeline = KneeOAPipeline()
    paths = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    rows = []
    for path in paths:
        source_bytes = path.read_bytes()
        source = cv2.imdecode(
            np.frombuffer(source_bytes, np.uint8), cv2.IMREAD_COLOR
        )
        knees = roi_service.detect_knees_with_coords(source_bytes)
        if not knees:
            knees = [
                {
                    "box": [0, 0, source.shape[1], source.shape[0]],
                    "crop_bytes": source_bytes,
                }
            ]
        knees = sorted(knees, key=lambda knee: knee["box"][0])

        for index, knee in enumerate(knees):
            side = side_for_roi(knee, source.shape[1], len(knees), index)
            tensor, processed, _ = preprocessing_service.preprocess_image(
                knee["crop_bytes"], knee_side=side
            )
            tensor = tensor.to(pipeline.device)
            with torch.inference_mode():
                outputs = {
                    name: model.forward_with_class_maps(tensor)
                    for name, model in pipeline.models.items()
                }
            logits = {name: output[0] for name, output in outputs.items()}
            probabilities = {
                name: F.softmax(value.float(), dim=1)
                for name, value in logits.items()
            }
            vote = weighted_soft_vote(logits, pipeline.ensemble_weights)
            grade = int(vote.argmax(dim=1).item())
            metrics = {}
            for name, model in pipeline.models.items():
                cam = native_cam_service.extract_cam(
                    model=model,
                    class_maps=outputs[name][1],
                    predicted_class=grade,
                    output_size=processed.shape[:2],
                )
                metrics[name] = native_cam_service.energy_metrics(cam)

            old_source = old_global_selector(probabilities, grade)
            new_source = select_heatmap_component(probabilities, grade, metrics)
            rows.append(
                {
                    "image": path.name,
                    "side": side,
                    "grade": grade,
                    "confidence": float(vote[0, grade].item()),
                    "old_source": old_source,
                    "new_source": new_source,
                    "selected_passed": passes(metrics[new_source]),
                    "old_selected_passed": passes(metrics[old_source]),
                    "metrics": metrics,
                    "class_support": {
                        name: float(value[0, grade].item())
                        for name, value in probabilities.items()
                    },
                }
            )

    changed = [row for row in rows if row["new_source"] != row["old_source"]]
    payload = {
        "labels_available": False,
        "image_count": len(paths),
        "roi_count": len(rows),
        "weights": pipeline.ensemble_weights,
        "grade_distribution": dict(
            sorted(Counter(row["grade"] for row in rows).items())
        ),
        "old_source_distribution": dict(
            Counter(row["old_source"] for row in rows)
        ),
        "new_source_distribution": dict(
            Counter(row["new_source"] for row in rows)
        ),
        "source_changed_count": len(changed),
        "old_selected_gate_pass_count": sum(
            row["old_selected_passed"] for row in rows
        ),
        "new_selected_gate_pass_count": sum(row["selected_passed"] for row in rows),
        "no_candidate_pass_count": sum(
            not any(passes(value) for value in row["metrics"].values())
            for row in rows
        ),
        "known_failure_rows": [
            row
            for row in rows
            if row["image"].split("_", 1)[0] in KNOWN_FAILURE_IDS
        ],
    }
    print("HEATMAP_AUDIT_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
