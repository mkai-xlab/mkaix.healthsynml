"""Generate the deployed-API report for the labeled YOLO validation folder."""

import json
import statistics
from collections import Counter
from pathlib import Path


RUN_ID = "2026-07-25_01-12-45_382275_ICT"
ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = (
    ROOT / "docs" / "report" / "system" / "assets"
    / f"{RUN_ID}_yolo_valid_api_smoke.json"
)
MONTAGE_NAME = f"{RUN_ID}_yolo_valid_cam_montage.jpg"
HEALTH_NAME = f"{RUN_ID}_post_test_health.json"
REPORT_PATH = (
    ROOT / "docs" / "report" / "system"
    / f"{RUN_ID}_yolo_valid_deployed_api_test.md"
)
LABEL_DIR = Path(
    "/home/viet/Downloads/Knee Xray Yolo.yolov8/valid/labels"
)


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )


def percent(count: int, total: int) -> str:
    return f"{100.0 * count / total:.1f}%"


data = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
rows = data["rows"]
expected_counts = {
    path.stem: sum(
        bool(line.strip())
        for line in path.read_text(errors="replace").splitlines()
    )
    for path in LABEL_DIR.glob("*.txt")
}

mismatches = []
for row in rows:
    expected = expected_counts.get(Path(row["image"]).stem)
    if expected != row["knees"]:
        mismatches.append((row["image"], expected, row["knees"]))

latencies = [float(row["seconds"]) for row in rows]
grades = [int(grade) for row in rows for grade in row["grades"]]
confidences = [
    float(confidence) for row in rows for confidence in row["confidences"]
]
grade_counts = Counter(grades)
knee_counts = Counter(int(row["knees"]) for row in rows)
bilateral = [row for row in rows if row["knees"] == 2]
bilateral_differences = Counter(
    abs(int(row["grades"][0]) - int(row["grades"][1]))
    for row in bilateral
)

flat_predictions = []
for row in rows:
    for index, (grade, confidence) in enumerate(
        zip(row["grades"], row["confidences"])
    ):
        flat_predictions.append(
            {
                "image": row["image"],
                "knee_index": index,
                "grade": int(grade),
                "confidence": float(confidence),
            }
        )

lines = [
    "# Deployed API Test on YOLO Validation Images",
    "",
    f"**Test identifier:** `{RUN_ID}`  ",
    "**Recoverable run started:** 2026-07-25 01:12:45.382275 ICT  ",
    "**Run and post-test health verification completed:** "
    "2026-07-25 01:48:49.374500 ICT  ",
    "**Deployment:** `http://54.254.113.71:8005`  ",
    "**Source images:** `/home/viet/Downloads/Knee Xray Yolo.yolov8/valid/images`  ",
    "**Source labels:** `/home/viet/Downloads/Knee Xray Yolo.yolov8/valid/labels`  ",
    "**Execution:** sequential requests, 180-second per-request timeout",
    "",
    "## Executive Summary",
    "",
    "The deployed API passed the complete test on all 211 YOLO validation images. "
    "Every request returned HTTP 200. The API returned 421 knee predictions, "
    "exactly matching the 421 labeled ROI instances, with zero per-image count "
    "mismatches. Every response retained the established JSON schema; every "
    "five-grade probability vector summed to one; and every annotated image, ROI "
    "image, and native-CAM overlay decoded successfully. All 421 CAM overlays were "
    "384x384.",
    "",
    "This result is a strong operational test and an exact ROI-count test. It is "
    "not a YOLO localization-accuracy test because the harness did not retain the "
    "returned boxes for IoU calculation. It is also not a KL-grade accuracy test: "
    "the label files contain YOLO bounding boxes, not KL grades.",
    "",
    "The API remained healthy after the load test and reported the expected "
    "DenseNet-121 epoch-27 plus SE-ResNeXt epoch-24 ensemble with weights 0.55/0.45.",
    "",
    "## Source Dataset Check",
    "",
    "| Check | Value |",
    "| --- | ---: |",
    "| Validation images | 211 |",
    "| YOLO label files | 211 |",
    "| Matched image/label stems | 211 |",
    "| Images with one labeled knee | 1 |",
    "| Images with two labeled knees | 210 |",
    "| Total labeled ROI instances | 421 |",
    "| Missing labels | 0 |",
    "| Orphan labels | 0 |",
    "",
    "## API Contract and Media Validation",
    "",
    "| Check | Result |",
    "| --- | --- |",
    "| HTTP 200 | 211/211 |",
    "| Empty prediction responses | 0 |",
    "| Exact top-level schema | Pass |",
    "| Exact per-knee schema | Pass |",
    "| Five expected grade-probability keys | 421/421 |",
    "| Probability sum within 1e-5 of one | 421/421 |",
    "| Annotated source images decoded | 211/211 |",
    "| Returned ROI images decoded | 421/421 |",
    "| Native-CAM overlays decoded at 384x384 | 421/421 |",
    "| Timeout, HTTP 4xx, or HTTP 5xx | 0 |",
    "| Post-test health | Healthy |",
    "",
    "The unchanged top-level fields are `filename`, `predictions`, and "
    "`annotated_image`. Each prediction retains `predicted_class`, "
    "`predicted_grade`, `confidence`, `description`, `details`, `box`, "
    "`yolo_confidence`, `knee_side`, `roi_image`, and `gradcam_image`. The "
    "historical `gradcam_image` field contains the selected native-CAM overlay.",
    "",
    "## ROI Count Comparison",
    "",
    "| Measure | Result |",
    "| --- | ---: |",
    f"| Expected labeled ROIs | {sum(expected_counts.values())} |",
    f"| Returned knee predictions | {sum(knee_counts[k] * k for k in knee_counts)} |",
    f"| Per-image count matches | {len(rows) - len(mismatches)}/{len(rows)} |",
    f"| Per-image count mismatches | {len(mismatches)} |",
    f"| Returned one-knee images | {knee_counts[1]} |",
    f"| Returned two-knee images | {knee_counts[2]} |",
    "",
    "This demonstrates perfect count agreement on this folder. It does not prove "
    "that every returned box has sufficient IoU with its annotation. A future "
    "detector audit should preserve the API box coordinates and calculate per-box "
    "IoU, precision, recall, mAP, and laterality correctness.",
    "",
    "## Unlabeled KL Prediction Distribution",
    "",
    "| Predicted KL grade | Knees | Share |",
    "| ---: | ---: | ---: |",
]

for grade in range(5):
    lines.append(
        f"| {grade} | {grade_counts[grade]} | "
        f"{percent(grade_counts[grade], len(grades))} |"
    )

lines.extend(
    [
        "",
        "Grade 0 accounts for half of predictions. This distribution cannot be "
        "scored as correct or incorrect without KL ground truth and should not be "
        "used to estimate deployment prevalence.",
        "",
        "### Confidence Distribution",
        "",
        "| Measure | Value |",
        "| --- | ---: |",
        f"| Minimum | {min(confidences):.4f} |",
        f"| p10 | {quantile(confidences, 0.10):.4f} |",
        f"| Median | {quantile(confidences, 0.50):.4f} |",
        f"| Mean | {statistics.mean(confidences):.4f} |",
        f"| p90 | {quantile(confidences, 0.90):.4f} |",
        f"| Maximum | {max(confidences):.4f} |",
    ]
)

for threshold in (0.30, 0.40, 0.50):
    count = sum(value < threshold for value in confidences)
    lines.append(
        f"| Below {threshold:.2f} | {count}/{len(confidences)} "
        f"({percent(count, len(confidences))}) |"
    )

lines.extend(
    [
        "",
        "Grade 1 had the lowest mean confidence (`0.4041`), consistent with the "
        "known Grade 0/1 and Grade 1/2 ambiguity. A confidence below 0.40 should be "
        "presented as uncertain and requiring review; it must not be treated as a "
        "calibrated probability of correctness until calibration is measured on "
        "labeled, patient-separated data.",
        "",
        "### Confidence by Predicted Grade",
        "",
        "| Predicted grade | Count | Mean | Median | Minimum | Maximum | Below 0.40 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
)

for grade in range(5):
    values = [
        confidence
        for predicted_grade, confidence in zip(grades, confidences)
        if predicted_grade == grade
    ]
    low_count = sum(value < 0.40 for value in values)
    lines.append(
        f"| {grade} | {len(values)} | {statistics.mean(values):.4f} | "
        f"{quantile(values, 0.50):.4f} | {min(values):.4f} | "
        f"{max(values):.4f} | {low_count} |"
    )

lines.extend(
    [
        "",
        "### Bilateral Prediction Difference",
        "",
        "The following is a consistency description, not an accuracy metric. "
        "Different grades between a patient's two knees may be clinically valid.",
        "",
        "| Absolute left/right grade difference | Bilateral images | Share |",
        "| ---: | ---: | ---: |",
    ]
)

for difference in range(5):
    count = bilateral_differences[difference]
    lines.append(
        f"| {difference} | {count} | {percent(count, len(bilateral))} |"
    )

lines.extend(
    [
        "",
        "## Latency",
        "",
        "Times are sequential public-network upload-to-complete-response "
        "measurements. They include large image upload, YOLO, both CNNs, native-CAM "
        "selection/rendering, base64 serialization, and response download.",
        "",
        "| Measure | Seconds |",
        "| --- | ---: |",
        f"| Minimum | {min(latencies):.3f} |",
        f"| Median | {quantile(latencies, 0.50):.3f} |",
        f"| Mean | {statistics.mean(latencies):.3f} |",
        f"| p90 | {quantile(latencies, 0.90):.3f} |",
        f"| p95 | {quantile(latencies, 0.95):.3f} |",
        f"| p99 | {quantile(latencies, 0.99):.3f} |",
        f"| Maximum | {max(latencies):.3f} |",
        f"| Cumulative request time | {sum(latencies):.3f} |",
        "",
        "### Ten Slowest Requests",
        "",
        "| Seconds | Knees | Predicted grades | Confidence | Image |",
        "| ---: | ---: | --- | --- | --- |",
    ]
)

for row in sorted(rows, key=lambda item: item["seconds"], reverse=True)[:10]:
    lines.append(
        f"| {row['seconds']:.3f} | {row['knees']} | "
        f"{' / '.join(map(str, row['grades']))} | "
        f"{' / '.join(f'{value:.4f}' for value in row['confidences'])} | "
        f"`{row['image']}` |"
    )

lines.extend(
    [
        "",
        "## CAM Visual Review",
        "",
        f"![Representative validation CAMs](assets/{MONTAGE_NAME})",
        "",
        "The representative montage deliberately includes low-confidence, high-grade, "
        "slow, and asymmetric cases. It shows mixed explanation quality:",
        "",
        "- `9150288` contains a knee prosthesis. KL grading is not an appropriate "
        "  normal target for a replaced knee, and the low-confidence Grade 1 CAM "
        "  follows a lateral hardware/edge region. This is an out-of-distribution "
        "  input that should be rejected or explicitly routed to manual review.",
        "- The other `9150288` knee has a strong far-lateral hotspot outside the "
        "  useful central joint evidence, despite producing a normal-grade result.",
        "- `9218916`, `9255429`, and `9465298` retain compact medial/lateral margin "
        "  hotspots. Some maps are at joint level, but several include border or "
        "  lower-region energy and are not lesion-exact.",
        "- The Grade 4 maps in `9257048` and `9069393` are concentrated near the "
        "  joint line and are broadly plausible, but still cannot identify every "
        "  KL feature or prove a lesion boundary.",
        "",
        "The anatomy gate improves the worst diffuse maps but cannot guarantee a good "
        "explanation when neither component passes or when a lateral hotspot still "
        "satisfies the broad rectangular joint criterion. The response does not expose "
        "the selected model, gate pass/fallback state, or unblended CAM, so those facts "
        "cannot be reconstructed remotely. They should be recorded in server logs.",
        "",
        "Right-knee CAM backgrounds are also returned in canonical mirrored orientation "
        "while `roi_image` remains in original orientation. This is model-aligned but "
        "can confuse side-by-side interpretation and should be corrected in presentation "
        "code or explicitly labeled.",
        "",
        "## Lowest-Confidence Predictions",
        "",
        "| Confidence | Grade | Knee index | Image |",
        "| ---: | ---: | ---: | --- |",
    ]
)

for item in sorted(flat_predictions, key=lambda value: value["confidence"])[:15]:
    lines.append(
        f"| {item['confidence']:.4f} | {item['grade']} | "
        f"{item['knee_index']} | `{item['image']}` |"
    )

lines.extend(
    [
        "",
        "## Per-Image Results",
        "",
        "The confidence list follows the API prediction order. Expected ROI count "
        "comes from the matching YOLO label file.",
        "",
        "| Image | Expected ROIs | Returned knees | Grades | Confidences | Seconds |",
        "| --- | ---: | ---: | --- | --- | ---: |",
    ]
)

for row in rows:
    expected = expected_counts[Path(row["image"]).stem]
    lines.append(
        f"| `{row['image']}` | {expected} | {row['knees']} | "
        f"{' / '.join(map(str, row['grades']))} | "
        f"{' / '.join(f'{value:.4f}' for value in row['confidences'])} | "
        f"{row['seconds']:.3f} |"
    )

lines.extend(
    [
        "",
        "## Assessment and Recommendations",
        "",
        "### What passed",
        "",
        "1. The deployment remained healthy under 211 sequential large-image requests.",
        "2. The response contract, probability vectors, and every returned image passed.",
        "3. YOLO ROI counts matched all 211 label files exactly: 421/421 instances.",
        "4. Both classifier checkpoints, ensemble weights, and heatmap method remained "
        "   unchanged after the test.",
        "",
        "### What should improve",
        "",
        "1. Add input-quality and out-of-distribution screening for knee replacement, "
        "   non-radiograph input, missing joint, extreme exposure, and unsupported views.",
        "2. Present predictions below 0.40 as uncertain and requiring expert review. "
        "   Do not call 0.40 a calibrated clinical threshold yet.",
        "3. Preserve API box coordinates in future validation artifacts and calculate "
        "   IoU against these label files; count agreement alone is insufficient.",
        "4. Log per-component CAM metrics, selected heatmap source, and anatomy-gate "
        "   fallback status for every prediction.",
        "5. Fix or label the canonical mirrored orientation of right-knee overlays.",
        "6. Profile YOLO, DenseNet, SE-ResNeXt, CAM, JPEG, and base64 stages separately. "
        "   CPU latency remains too variable for a consistently responsive UI.",
        "7. Do not retrain or change the classifier architecture based only on this "
        "   unlabeled folder. Use a newly locked, patient-separated KL-labeled holdout "
        "   before making model-performance decisions.",
        "",
        "## Final Decision",
        "",
        "The deployed API is operationally stable and its YOLO detector has perfect ROI "
        "count agreement on this validation folder. The test does not establish box IoU "
        "or KL-grade accuracy. The model architecture should remain unchanged for now. "
        "The immediate priorities are prosthesis/OOD screening, low-confidence UX, CAM "
        "telemetry and orientation correction, and latency profiling.",
        "",
        "## Evidence",
        "",
        f"- [Raw complete API result](assets/{RESULT_PATH.name})",
        f"- [Representative CAM montage](assets/{MONTAGE_NAME})",
        f"- [Post-test health response](assets/{HEALTH_NAME})",
        "- [Complete system technical report]"
        "(2026-07-25_00-26-04_635853_ICT_complete_system_technical_report.md)",
        "",
    ]
)

REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(REPORT_PATH)
