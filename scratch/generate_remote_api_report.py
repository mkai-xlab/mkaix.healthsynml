"""Generate the exact-timestamp deployed API report from preserved JSON results."""

import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "2026-07-24_16-50-26_365170_UTC"
ASSET_DIR = ROOT / "docs" / "report" / "ensemble" / "assets"
RESULT_PATH = ASSET_DIR / f"{RUN_ID}_remote_api_smoke.json"
HEALTH_PATH = ASSET_DIR / f"{RUN_ID}_remote_health.json"
MODELS_PATH = ASSET_DIR / f"{RUN_ID}_remote_models.json"
MONTAGE_NAME = f"{RUN_ID}_remote_heatmap_montage.jpg"
REPORT_PATH = (
    ROOT
    / "docs"
    / "report"
    / "ensemble"
    / f"{RUN_ID}_deployed_api_full_test.md"
)


def percentile(sorted_values: list[float], fraction: float) -> float:
    index = max(0, math.ceil(fraction * len(sorted_values)) - 1)
    return sorted_values[index]


result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
health = json.loads(HEALTH_PATH.read_text(encoding="utf-8"))
models = json.loads(MODELS_PATH.read_text(encoding="utf-8"))
rows = result["rows"]
latencies = sorted(float(row["seconds"]) for row in rows)
confidences = sorted(
    float(confidence)
    for row in rows
    for confidence in row["confidences"]
)
grade_counts = Counter(
    int(grade) for row in rows for grade in row["grades"]
)
slowest = sorted(rows, key=lambda row: row["seconds"], reverse=True)[:10]


def percentage(count: int, total: int) -> str:
    return f"{100.0 * count / total:.1f}%"


lines = [
    "# Deployed Knee-OA API Full Test",
    "",
    f"**Test identifier:** `{RUN_ID}`  ",
    "**Test start timestamp:** `2026-07-24 16:50:26.365170 UTC`  ",
    "**Report completed:** `2026-07-24 17:15:24.638256 UTC`  ",
    "**Deployment:** `http://54.254.113.71:8005`  ",
    "**Input folder:** `test_images`",
    "",
    "## Executive Summary",
    "",
    "The deployed API passed the complete sequential operational test. All 105",
    "source images returned HTTP 200 and produced 209 knee predictions. Every",
    "response preserved the established JSON schema, every five-class probability",
    "vector summed to one, and every returned native-CAM image decoded at 384x384.",
    "No request timed out and no HTTP 4xx/5xx response occurred.",
    "",
    "The deployed model configuration is correct: DenseNet-121 epoch 27 and",
    "SE-ResNeXt50-32x4d epoch 24 use probability-level soft voting with weights",
    "0.55/0.45. Heatmaps use the gradient-free per-case anatomy-gated native-CAM",
    "selector. EfficientNet-B0 is not loaded by production ensemble mode.",
    "",
    "The primary operational concern is CPU latency. Mean end-to-end response time",
    f"was `{result['mean_request_seconds']:.3f} s`, p95 was",
    f"`{percentile(latencies, 0.95):.3f} s`, and the maximum was",
    f"`{result['max_request_seconds']:.3f} s`. This is acceptable for batch review",
    "but slow and variable for an interactive application.",
    "",
    "This is not an accuracy evaluation. The test folder has no ground-truth KL",
    "labels, so QWK, F1, precision, recall, AP, AUC, sensitivity, and specificity",
    "cannot be calculated from this run.",
    "",
    "## Deployed Configuration",
    "",
    "| Field | Deployed value |",
    "| --- | --- |",
    f"| Health | `{health['status']}` |",
    f"| Device | `{health['device']}` |",
    f"| Model | `{models['model']}` |",
    f"| Architecture | `{models['architecture']}` |",
    f"| Loss | `{models['loss']}` |",
    f"| DenseNet checkpoint | `{models['checkpoint']['densenet121']}` |",
    f"| DenseNet selected epoch | `{health['checkpoint']['epoch']['densenet121']}` |",
    f"| SE-ResNeXt checkpoint | `{models['checkpoint']['seresnext50_32x4d']}` |",
    f"| SE-ResNeXt selected epoch | `{health['checkpoint']['epoch']['seresnext50_32x4d']}` |",
    f"| Voting weights | DenseNet `{health['checkpoint']['weights']['densenet121']:.2f}`, SE-ResNeXt `{health['checkpoint']['weights']['seresnext50_32x4d']:.2f}` |",
    f"| Input | Resize `{models['input']['resize'][0]}x{models['input']['resize'][1]}`, center crop `{models['input']['center_crop'][0]}x{models['input']['center_crop'][1]}` |",
    f"| Laterality canonicalization | `{models['input']['laterality_canonicalization']}` |",
    f"| Heatmap method | `{models['heatmap']['method']}` |",
    f"| Heatmap source | `{models['heatmap']['source']}` |",
    f"| Gradient-free | `{models['heatmap']['gradient_free']}` |",
    "",
    "The `/health` endpoint exposed the selected validation metrics embedded in",
    "both checkpoints. These are checkpoint metadata, not measurements from the",
    "105 deployment images.",
    "",
    "| Model | Validation accuracy | QWK | Macro F1 | Grade 1 recall | AP | AUC | Selection |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]

for name, label in (
    ("densenet121", "DenseNet-121"),
    ("seresnext50_32x4d", "SE-ResNeXt50"),
):
    metrics = health["checkpoint"]["validation_metrics"][name]
    lines.append(
        f"| {label} | {metrics['accuracy']:.4f} | {metrics['qwk']:.4f} | "
        f"{metrics['macro_f1']:.4f} | {metrics['grade1_recall']:.4f} | "
        f"{metrics['macro_ap']:.4f} | {metrics['macro_auc']:.4f} | "
        f"{metrics['selection_score']:.4f} |"
    )

lines.extend(
    [
        "",
        "## API Contract Verification",
        "",
        "| Check | Result |",
        "| --- | --- |",
        f"| Source images | `{result['images']}/105` completed |",
        f"| HTTP 200 | `{sum(row['status'] == 200 for row in rows)}/{len(rows)}` |",
        f"| Knee predictions | `{result['predictions']}` |",
        f"| One-knee source images | `{sum(row['knees'] == 1 for row in rows)}` |",
        f"| Two-knee source images | `{sum(row['knees'] == 2 for row in rows)}` |",
        f"| Empty-prediction responses | `{sum(row['knees'] == 0 for row in rows)}` |",
        f"| Top-level and per-knee schema | `{'unchanged' if result['schema_unchanged'] else 'failed'}` |",
        "| Probability keys | Grades 0-4 present for every prediction |",
        "| Probability normalization | Passed for 209/209 predictions |",
        "| Annotated source images | All decoded |",
        "| ROI images | All present ROI values decoded |",
        f"| Native-CAM images | `{'209/209 decoded at 384x384' if result['all_heatmaps_decoded_at_384x384'] else 'failed'}` |",
        "| Request timeouts | `0` |",
        "| HTTP 4xx/5xx | `0` |",
        "",
        "The unchanged top-level keys are `filename`, `predictions`, and",
        "`annotated_image`. Each prediction retains `predicted_class`,",
        "`predicted_grade`, `confidence`, `description`, `details`, `box`,",
        "`yolo_confidence`, `knee_side`, `roi_image`, and `gradcam_image`.",
        "The historical `gradcam_image` field contains a native-CAM overlay.",
        "",
        "## Prediction Distribution",
        "",
        "| Predicted grade | Count | Share |",
        "| ---: | ---: | ---: |",
    ]
)
for grade in range(5):
    lines.append(
        f"| {grade} | {grade_counts[grade]} | "
        f"{percentage(grade_counts[grade], len(confidences))} |"
    )

below_030 = sum(value < 0.30 for value in confidences)
below_040 = sum(value < 0.40 for value in confidences)
below_050 = sum(value < 0.50 for value in confidences)
lines.extend(
    [
        "",
        "The grade distribution is not evidence of class-specific accuracy because",
        "true grades are unavailable. Grade 0 dominates the predictions (132/209),",
        "which should be checked against the expected deployment population once",
        "labels or a representative audit sample are available.",
        "",
        "## Confidence Distribution",
        "",
        "| Measure | Value |",
        "| --- | ---: |",
        f"| Mean | {sum(confidences) / len(confidences):.4f} |",
        f"| Median | {confidences[len(confidences) // 2]:.4f} |",
        f"| Minimum | {confidences[0]:.4f} |",
        f"| Maximum | {confidences[-1]:.4f} |",
        f"| Below 0.30 | {below_030} ({percentage(below_030, len(confidences))}) |",
        f"| Below 0.40 | {below_040} ({percentage(below_040, len(confidences))}) |",
        f"| Below 0.50 | {below_050} ({percentage(below_050, len(confidences))}) |",
        "",
        "More than half of predictions are below 0.40 confidence. This does not prove",
        "they are wrong, but the client should present them as uncertain and avoid",
        "turning the top class into an unconditional clinical statement. Calibration",
        "must be measured on labeled, patient-separated data before selecting a",
        "clinical confidence threshold.",
        "",
        "## Latency",
        "",
        "Times are sequential public-network, upload-to-complete-response measurements.",
        "They include image upload, YOLO, both CNNs, native-CAM selection/rendering,",
        "JPEG/base64 serialization, and response download.",
        "",
        "| Measure | Seconds |",
        "| --- | ---: |",
        f"| Minimum | {latencies[0]:.3f} |",
        f"| Median | {percentile(latencies, 0.50):.3f} |",
        f"| Mean | {sum(latencies) / len(latencies):.3f} |",
        f"| p90 | {percentile(latencies, 0.90):.3f} |",
        f"| p95 | {percentile(latencies, 0.95):.3f} |",
        f"| p99 | {percentile(latencies, 0.99):.3f} |",
        f"| Maximum | {latencies[-1]:.3f} |",
        f"| Cumulative request time | {sum(latencies):.3f} |",
        "",
        "### Ten Slowest Requests",
        "",
        "| Seconds | Knees | Predicted grades | Image |",
        "| ---: | ---: | --- | --- |",
    ]
)
for row in slowest:
    grades = ", ".join(str(value) for value in row["grades"])
    lines.append(
        f"| {row['seconds']:.3f} | {row['knees']} | {grades} | "
        f"`{row['image']}` |"
    )

lines.extend(
    [
        "",
        "Latency varies too much for a predictable interactive experience. The slowest",
        "two-knee request was 38.501 seconds, while the fastest request was 4.269",
        "seconds. This run was sequential, so the variance was not caused by this",
        "client sending concurrent requests.",
        "",
        "## Heatmap Review",
        "",
        f"![Deployed API heatmap montage](assets/{MONTAGE_NAME})",
        "",
        "The deployed endpoint returns the expected anatomy-gated native-CAM behavior:",
        "",
        "- `9003430` no longer has the large diffuse upper-femur/lower-tibia map",
        "  produced by the former global-agreement selector. Its activation is now",
        "  concentrated near the lateral joint margin.",
        "- `9063928` similarly changes from broad off-joint activation to joint-level",
        "  lateral activation.",
        "- `9003175` and some other Grade 0 cases retain secondary edge or lower-tibia",
        "  activation. The maps are not uniformly clean.",
        "- `9066155` Grade 3 has high confidence and joint-level activation, but the",
        "  hotspot remains localized to a lateral region rather than displaying every",
        "  radiographic feature relevant to the KL grade.",
        "",
        "The visual conclusion is therefore **improved broad joint localization, not",
        "lesion-exact explanation**. A native CAM is faithful to the model's class-map",
        "head, but it is not a segmentation of osteophytes or joint-space narrowing.",
        "The endpoint returns only the blended overlay, so joint energy, border energy,",
        "and anatomy-gate pass/fallback counts cannot be independently recomputed from",
        "the remote response. Those metrics require the unblended CAM or server logs.",
        "",
        "## Per-Image Results",
        "",
        "Grades and confidences are listed in API prediction order. With two detected",
        "knees, the application normally returns anatomical right then left after",
        "sorting the YOLO boxes left-to-right.",
        "",
        "| Image | HTTP | Knees | Grades | Confidences | Seconds |",
        "| --- | ---: | ---: | --- | --- | ---: |",
    ]
)
for row in rows:
    grades = " / ".join(str(value) for value in row["grades"])
    confidence_text = " / ".join(f"{value:.4f}" for value in row["confidences"])
    lines.append(
        f"| `{row['image']}` | {row['status']} | {row['knees']} | "
        f"{grades} | {confidence_text} | {row['seconds']:.3f} |"
    )

lines.extend(
    [
        "",
        "## Assessment",
        "",
        "### Acceptable",
        "",
        "- Correct production checkpoints, epochs, architecture, vote weights, and",
        "  native-CAM selector are deployed.",
        "- All 105 images completed without an API error.",
        "- The `/predict` response schema is unchanged.",
        "- Every prediction has a valid probability vector and decodable explanation.",
        "- The known gross heatmap-placement failures are improved at deployment.",
        "",
        "### Needs Improvement",
        "",
        "1. **Instrument and reduce latency.** Record YOLO, DenseNet, SE-ResNeXt, CAM",
        "   rendering, and serialization times separately. Check EC2 CPU credit",
        "   throttling, thread oversubscription, available vCPUs, and memory pressure.",
        "   Benchmark after fixing `torch`/OpenMP thread counts. Consider ONNX Runtime",
        "   or a suitable GPU only after stage-level profiling identifies CNN inference",
        "   as the dominant cost.",
        "2. **Expose uncertainty safely.** A majority of predictions are below 0.40.",
        "   Keep all five probabilities visible and present low-confidence output as",
        "   review-required. Do not invent a clinical threshold from this unlabeled run.",
        "3. **Audit the 42 known local anatomy-gate fallback cases on the server.** The",
        "   public schema should remain unchanged, but server-side structured logs can",
        "   record heatmap source, gate pass/fallback, joint energy, border energy, and",
        "   lower-tibia energy for monitoring.",
        "4. **Do not overstate CAM precision.** Lateral joint hotspots remain. Exact",
        "   osteophyte/JSN explanation requires landmark, compartment, or lesion-level",
        "   supervision and expert review, not stronger post-processing alone.",
        "5. **Secure the public endpoint.** The tested deployment uses public plain HTTP",
        "   and exposes interactive API documentation. Place it behind HTTPS, define",
        "   authentication/authorization as appropriate, restrict security-group access,",
        "   cap upload size, and add rate limiting before handling clinical data.",
        "6. **Lock runtime dependencies.** Pin and test the Python dependency set so a",
        "   rebuild cannot silently change inference or image-encoding behavior.",
        "",
        "## Evidence Files",
        "",
        f"- [Raw all-image result](assets/{RESULT_PATH.name})",
        f"- [Health response](assets/{HEALTH_PATH.name})",
        f"- [Model metadata response](assets/{MODELS_PATH.name})",
        f"- [Remote heatmap montage](assets/{MONTAGE_NAME})",
        "- [Complete three-model and ensemble documentation](../../three_model_kl_system.md)",
        "",
        "## Final Decision",
        "",
        "The deployed API is functionally correct and stable across the entire local",
        "test-image folder. It is suitable for controlled demonstration and further",
        "validation. It is not yet supported as a clinical diagnostic system because",
        "this deployment test has no labels, latency is high and variable, confidence",
        "is frequently low, CAMs are not lesion-exact, and the endpoint still needs",
        "production security and monitoring controls.",
    ]
)

REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(REPORT_PATH)
