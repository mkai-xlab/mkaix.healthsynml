"""Compare deployment voting weights on unlabeled app smoke-test images."""

import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from app.ml.pipelines.knee_oa_pipeline import (
    KneeOAPipeline,
    select_heatmap_component,
    weighted_soft_vote,
)
from app.services.preprocessing_service import preprocessing_service
from app.services.roi_service import roi_service


IMAGE_DIR = Path("/test_images")
SAMPLE_SIZE = 20
SAMPLE_SEED = 20260724
TRIALS = {
    "equal": {
        "densenet121": 1 / 3,
        "seresnext50_32x4d": 1 / 3,
        "efficientnet_b0": 1 / 3,
    },
    "user_example_50_30_20": {
        "densenet121": 0.50,
        "seresnext50_32x4d": 0.30,
        "efficientnet_b0": 0.20,
    },
    "selected_50_35_15": {
        "densenet121": 0.50,
        "seresnext50_32x4d": 0.35,
        "efficientnet_b0": 0.15,
    },
    "dense_se_only_55_45_0": {
        "densenet121": 0.55,
        "seresnext50_32x4d": 0.45,
        "efficientnet_b0": 0.00,
    },
    "dense_dominant_60_30_10": {
        "densenet121": 0.60,
        "seresnext50_32x4d": 0.30,
        "efficientnet_b0": 0.10,
    },
}


def entropy(probabilities: torch.Tensor) -> float:
    values = probabilities.clamp_min(1e-12)
    return float(-(values * values.log()).sum().item())


def select_side(knee: dict, image_width: int, knee_count: int, index: int) -> str:
    if knee_count == 2:
        return ("right", "left")[index]
    center_x = (knee["box"][0] + knee["box"][2]) / 2
    if center_x < image_width * 0.40:
        return "right"
    if center_x > image_width * 0.60:
        return "left"
    return "unknown"


def main() -> None:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    pipeline = KneeOAPipeline()
    image_paths = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    selected_paths = sorted(
        random.Random(SAMPLE_SEED).sample(image_paths, min(SAMPLE_SIZE, len(image_paths)))
    )

    cases = []
    for image_path in selected_paths:
        image_bytes = image_path.read_bytes()
        # Decode only to reproduce the API's single-knee side heuristic.
        source = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        knees = roi_service.detect_knees_with_coords(image_bytes)
        if not knees:
            knees = [{"box": [0, 0, source.shape[1], source.shape[0]], "crop_bytes": image_bytes}]

        for index, knee in enumerate(knees):
            side = select_side(knee, source.shape[1], len(knees), index)
            tensor, _, _ = preprocessing_service.preprocess_image(
                knee["crop_bytes"], knee_side=side
            )
            tensor = tensor.to(pipeline.device)
            with torch.no_grad():
                logits = {
                    name: model(tensor) for name, model in pipeline.models.items()
                }
            probabilities = {
                name: F.softmax(value.float(), dim=1)
                for name, value in logits.items()
            }
            cases.append(
                {
                    "image": image_path.name,
                    "roi_index": index,
                    "side": side,
                    "logits": logits,
                    "probabilities": probabilities,
                    "component_grades": {
                        name: int(value.argmax(dim=1).item())
                        for name, value in probabilities.items()
                    },
                }
            )

    summaries = {}
    trial_predictions = {}
    for trial_name, weights in TRIALS.items():
        predictions = []
        confidences = []
        entropies = []
        majority_matches = 0
        changes_from_dense = 0
        for case in cases:
            vote = weighted_soft_vote(case["logits"], weights)
            grade = int(vote.argmax(dim=1).item())
            predictions.append(grade)
            confidences.append(float(vote[0, grade].item()))
            entropies.append(entropy(vote[0]))
            component_counts = Counter(case["component_grades"].values())
            majority_grade, majority_count = component_counts.most_common(1)[0]
            majority_matches += int(majority_count >= 2 and grade == majority_grade)
            changes_from_dense += int(
                grade != case["component_grades"]["densenet121"]
            )

        summaries[trial_name] = {
            "weights": weights,
            "case_count": len(cases),
            "grade_distribution": dict(sorted(Counter(predictions).items())),
            "mean_confidence": sum(confidences) / len(confidences),
            "mean_entropy": sum(entropies) / len(entropies),
            "majority_matches": majority_matches,
            "changes_from_densenet": changes_from_dense,
        }
        trial_predictions[trial_name] = predictions

    selected_rows = []
    selected_weights = TRIALS["selected_50_35_15"]
    for case in cases:
        vote = weighted_soft_vote(case["logits"], selected_weights)
        grade = int(vote.argmax(dim=1).item())
        heatmap_source = select_heatmap_component(
            case["probabilities"], grade, pipeline.localization_scores
        )
        selected_rows.append(
            {
                "image": case["image"],
                "roi_index": case["roi_index"],
                "side": case["side"],
                "densenet_grade": case["component_grades"]["densenet121"],
                "se_resnext_grade": case["component_grades"]["seresnext50_32x4d"],
                "efficientnet_b0_grade": case["component_grades"]["efficientnet_b0"],
                "ensemble_grade": grade,
                "confidence": float(vote[0, grade].item()),
                "heatmap_source": heatmap_source,
            }
        )

    payload = {
        "sample_seed": SAMPLE_SEED,
        "selected_image_count": len(selected_paths),
        "roi_case_count": len(cases),
        "labels_available": False,
        "component_grade_distributions": {
            name: dict(
                sorted(Counter(case["component_grades"][name] for case in cases).items())
            )
            for name in pipeline.models
        },
        "unanimous_component_cases": sum(
            len(set(case["component_grades"].values())) == 1 for case in cases
        ),
        "trial_summaries": summaries,
        "selected_rows": selected_rows,
    }
    print("RESULT_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
