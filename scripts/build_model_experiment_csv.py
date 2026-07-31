"""Build normalized CSV indexes from the two model execution reports."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "docs" / "report"

FIELDS = [
    "model_family", "run_timestamp", "timezone", "record_type", "status",
    "short_description", "purpose", "decision", "known_issue",
    "source_report", "notebook_archive", "checkpoint_directory",
    "architecture", "initialization", "input_size", "roi_and_crop_policy",
    "image_processing_full", "training_augmentation_full",
    "validation_test_processing", "laterality_policy", "sampler",
    "pipeline", "training_stages", "epochs", "selected_epoch", "batch_size",
    "workers_gpu", "loss_function", "compared_arms", "optimizer", "scheduler",
    "learning_rates", "weight_decay", "amp", "gradient_clipping", "ema",
    "seed", "checkpoint_selection", "heatmap_method", "heatmap_resolution",
    "dataset_split", "test_n", "test_accuracy", "test_qwk", "test_mae",
    "test_macro_precision", "test_macro_recall", "test_macro_f1",
    "test_grade1_precision", "test_grade1_recall", "test_average_precision",
    "test_roc_auc", "test_loss", "test_composite", "confidence_interval_notes",
    "validation_accuracy", "validation_qwk", "validation_macro_f1",
    "validation_grade1_recall", "validation_average_precision",
    "validation_roc_auc", "validation_selection_score", "best_validation_epoch",
    "cam_cases", "cam_joint_energy", "cam_border_energy",
    "cam_lower_tibia_energy", "cam_peak_inside_rate",
    "cam_occlusion_metric", "cam_flip_correlation", "cam_notes",
    "validation_failures", "boundary_confusions", "critical_under_predictions",
    "critical_over_predictions", "experiment_results", "notes",
]


def clean(value: str) -> str:
    value = re.sub(r"[`*_]", "", value)
    value = re.sub(r"<br\s*/?>", "; ", value)
    value = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", value)
    return re.sub(r"\s+", " ", value).strip()


def point(value: str) -> str:
    match = re.search(r"(?<![\d-])(?:0|1)\.\d+|(?<![\d-])\d+\.\d+", clean(value))
    return match.group(0) if match else "not reported"


def table_after(section: str, heading: str) -> list[list[str]]:
    match = re.search(rf"^###? {re.escape(heading)}\s*$", section, re.M)
    if not match:
        return []
    tail = section[match.end():]
    next_heading = re.search(r"^#{2,4} ", tail, re.M)
    block = tail[:next_heading.start()] if next_heading else tail
    rows = []
    for line in block.splitlines():
        if not line.startswith("|"):
            continue
        cells = [clean(cell) for cell in line.strip().strip("|").split("|")]
        if cells and not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
            rows.append(cells)
    return rows


def key_values(rows: list[list[str]]) -> dict[str, str]:
    return {row[0]: row[1] for row in rows[1:] if len(row) >= 2}


def summary_paragraph(section: str) -> str:
    match = re.search(r"^### Summary\s*$", section, re.M)
    if not match:
        return "not reported"
    tail = section[match.end():]
    next_heading = re.search(r"^### ", tail, re.M)
    block = tail[:next_heading.start()] if next_heading else tail
    paragraphs = [clean(p) for p in re.split(r"\n\s*\n", block) if clean(p)]
    return paragraphs[0] if paragraphs else "not reported"


def notebook_operations(path: Path) -> str:
    if not path.exists():
        return "not reported"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    operations = []
    patterns = [
        r"(?:OpenCV)?CLAHE\((?!object)[^\n]*\)", r"(?:SquarePad|PadToSquare)\([^\n]*\)",
        r"transforms\.(?:RandomHorizontalFlip|RandomVerticalFlip|RandomRotation|ColorJitter|Resize|CenterCrop|RandomCrop|RandomResizedCrop|RandomErasing|GaussianBlur|Normalize|ToTensor)\([^\n]*",
        r"RandomGammaCorrection\([^\n]*\)", r"AddGaussianNoise\([^\n]*\)",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, source):
            item = clean(match.rstrip(","))
            if item not in operations:
                operations.append(item)
    return "; ".join(operations) if operations else "not reported"


def notebook_hyperparameters(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    result = {}
    for name in (
        "batch_size", "seed", "lr_warmup", "lr_coarse_head",
        "lr_coarse_backbone", "lr_finetune", "lr_standard", "weight_decay",
        "scheduler_stage2", "scheduler_stage3", "scheduler_standard",
    ):
        match = re.search(rf"^\s*{name}\s*=\s*([^#\n]+)", source, re.M)
        if match:
            result[name] = clean(match.group(1).rstrip(","))
    return result


def classification_metrics(section: str) -> dict[str, str]:
    result = {}
    macro = re.search(
        r"^\s*macro avg\s+(0\.\d+)\s+(0\.\d+)\s+(0\.\d+)\s+\d+",
        section, re.M,
    )
    if macro:
        result["precision"], result["recall"], result["f1"] = macro.groups()
    grade1 = re.search(
        r"^\s*(?:Grade\s*)?1\s+(0\.\d+)\s+(0\.\d+)\s+(0\.\d+)\s+\d+",
        section, re.M,
    )
    if grade1:
        result["grade1_precision"], result["grade1_recall"] = grade1.groups()[:2]
    return result


def top_summary(report: str) -> dict[str, dict[str, str]]:
    rows = []
    for line in report.splitlines():
        if line.startswith("| 2026-"):
            cells = [clean(cell) for cell in line.strip().strip("|").split("|")]
            rows.append(cells)
        elif rows:
            break
    result = {}
    for cells in rows:
        if len(cells) >= 6:
            timestamp = cells[0].replace(".633270", ".633270").strip()
            result[timestamp[:19]] = {
                "description": cells[1], "accuracy": point(cells[2]),
                "qwk": point(cells[3]), "auc": point(cells[4]), "ap": point(cells[5]),
                "extra": cells[6:], "ci": "; ".join(cells[2:6]),
            }
    return result


def find_notebook(directory: Path, timestamp: str) -> Path | None:
    prefix = timestamp[:19].replace(" ", "_").replace(":", "-")
    matches = sorted(directory.glob(f"{prefix}*.ipynb"))
    return matches[0] if matches else None


def parse_model(directory_name: str, model_family: str) -> list[dict[str, str]]:
    directory = REPORT_ROOT / directory_name
    report_path = directory / "report.md"
    report = report_path.read_text(encoding="utf-8")
    summary = top_summary(report)
    starts = list(re.finditer(r"^## Run: ([^\n]+)$", report, re.M))
    records: list[dict[str, str]] = []

    for index, start in enumerate(starts):
        section = report[start.start(): starts[index + 1].start() if index + 1 < len(starts) else len(report)]
        heading = clean(start.group(1))
        timestamp_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)", heading)
        if not timestamp_match:
            continue
        timestamp = timestamp_match.group(1)
        description_match = re.search(r"\((.+)\)\s*$", heading)
        description = clean(description_match.group(1)) if description_match else heading
        config = key_values(table_after(section, "Configurations"))
        metrics = key_values(table_after(section, "Final Test Metrics"))
        if not metrics:
            metrics = key_values(table_after(section, "Final Labeled Test Metrics"))
        validation = key_values(table_after(section, "Selected Validation Metrics"))
        if not validation:
            validation = key_values(table_after(section, "Validation Metrics at Selected Epoch"))
        top = summary.get(timestamp[:19], {})
        notebook = find_notebook(directory, timestamp)
        if notebook is None and timestamp.startswith("2026-07-25 06:30:25"):
            candidate = directory / "dense_net_121.ipynb"
            notebook = candidate if candidate.exists() else None
        notebook_config = notebook_hyperparameters(notebook)
        class_metrics = classification_metrics(section)
        augmentation_parts = []
        for key in (
            "Random Horizontal Flip", "Minority Augmentations", "Test-Time Augmentation",
            "Minority Augmentations / TTA", "Random Erasing", "Gamma / Brightness / Contrast",
            "Rotation / Random Erasing", "Gamma / Gaussian Noise",
        ):
            if key in config:
                augmentation_parts.append(f"{key}: {config[key]}")

        row = {field: "not reported" for field in FIELDS}
        row.update({
            "model_family": model_family,
            "run_timestamp": timestamp,
            "timezone": "UTC" if "UTC" in heading else "timezone not recorded",
            "record_type": "training run",
            "status": "rejected" if "REJECTED" in heading else ("selected/deployed for external audit" if "SELECTED" in heading else ("historical production" if "PRODUCTION" in heading else "completed")),
            "short_description": description,
            "purpose": summary_paragraph(section),
            "source_report": str(report_path.relative_to(ROOT)),
            "notebook_archive": str(notebook.relative_to(ROOT)) if notebook else "not archived in this model report folder",
            "checkpoint_directory": config.get("Checkpoint Directory", config.get("Run Directory", "not reported")),
            "architecture": config.get("Architecture", config.get("Checkpoint Architecture", "standard classifier head; exact architecture label not reported")),
            "initialization": config.get("Model", "ImageNet-pretrained backbone; exact initialization not reported"),
            "input_size": config.get("Model Input", config.get("Image Size", "not reported")),
            "roi_and_crop_policy": config.get("Input Crop", "derived from Model Input; exact ROI policy not reported"),
            "image_processing_full": config.get("Input Preprocessing", notebook_operations(notebook) if notebook else "not reported"),
            "training_augmentation_full": config.get("Training Augmentation", "; ".join(augmentation_parts) or (notebook_operations(notebook) if notebook else "not reported")),
            "validation_test_processing": config.get("Validation / Test Transform", "deterministic validation/test transform; exact list not reported"),
            "laterality_policy": config.get("Orientation Policy", config.get("Laterality Canonicalization", "not reported")),
            "sampler": config.get("Sampler", config.get("Balanced Sampler", "not reported")),
            "pipeline": config.get("Pipeline", "not reported"),
            "training_stages": config.get("Training Schedule", config.get("Epochs", config.get("Configured Epochs", "not reported"))),
            "epochs": config.get("Epochs", config.get("Configured Epochs", config.get("Training Schedule", "not reported"))),
            "selected_epoch": config.get("Selected Loss / Epoch", config.get("Selected Checkpoint", config.get("Completed / Selected Epoch", "not reported"))),
            "batch_size": config.get("Batch Size", config.get("Batch Size / Workers", config.get("Batch Size / Workers / GPU", config.get("Batch Size / AMP", config.get("Batch Size / AMP / GPU", notebook_config.get("batch_size", "not reported")))))),
            "workers_gpu": config.get("Batch Size / Workers / GPU", config.get("Batch Size / AMP / GPU", "not reported")),
            "loss_function": config.get("Loss Function", config.get("Compared Loss Arms", top.get("description", "not reported"))),
            "compared_arms": config.get("Compared Loss Arms", "not applicable"),
            "optimizer": "AdamW" if "AdamW" in section else "not reported",
            "scheduler": config.get("Scheduler", "; ".join(f"{k}={v}" for k, v in notebook_config.items() if k.startswith("scheduler_")) or "not reported"),
            "learning_rates": "; ".join(f"{key}: {value}" for key, value in config.items() if "Learning Rate" in key or "Learning Rates" in key) or "; ".join(f"{k}={v}" for k, v in notebook_config.items() if k.startswith("lr_")) or "not reported",
            "weight_decay": config.get("Weight Decay", notebook_config.get("weight_decay", "not reported")),
            "amp": config.get("AMP / Gradient Clipping", config.get("Batch Size / AMP", "not reported")),
            "gradient_clipping": config.get("AMP / Gradient Clipping", "not reported"),
            "ema": config.get("EMA", config.get("EMA Decay", "disabled or not reported")),
            "seed": config.get("Seed", notebook_config.get("seed", "not reported")),
            "checkpoint_selection": config.get("Checkpoint Selection", config.get("Selected Checkpoint", "not reported")),
            "heatmap_method": config.get("Native-CAM Head", config.get("Classifier / Explanation Head", config.get("Archived Grad-CAM Method", "Grad-CAM or CAM method not reported"))),
            "heatmap_resolution": "12x12" if "12x12" in section and "24x24" not in description else ("24x24 + 12x12 fusion" if "24x24" in description else "not reported"),
            "dataset_split": config.get("Dataset Sizes", "historical Kaggle train/validation/test split; patient grouping not documented"),
            "test_n": "1656" if "1656" in section else "not reported",
            "test_accuracy": point(metrics.get("Accuracy", top.get("accuracy", ""))),
            "test_qwk": point(metrics.get("QWK Score", top.get("qwk", ""))),
            "test_mae": point(metrics.get("MAE", "")),
            "test_macro_precision": point(metrics.get("Macro Precision", class_metrics.get("precision", ""))),
            "test_macro_recall": point(metrics.get("Macro Recall", class_metrics.get("recall", ""))),
            "test_macro_f1": point(metrics.get("Macro F1", class_metrics.get("f1", ""))),
            "test_grade1_precision": point(metrics.get("Grade 1 Precision", class_metrics.get("grade1_precision", ""))),
            "test_grade1_recall": point(metrics.get("Grade 1 Recall", class_metrics.get("grade1_recall", ""))),
            "test_average_precision": point(metrics.get("Average Precision", top.get("ap", ""))),
            "test_roc_auc": point(metrics.get("ROC AUC", top.get("auc", ""))),
            "test_loss": point(metrics.get("Loss", "")),
            "test_composite": point(metrics.get("Composite Score", metrics.get("Composite score, reported only", ""))),
            "confidence_interval_notes": top.get("ci", "see source report; CI not reported in summary"),
            "validation_accuracy": point(validation.get("Accuracy", "")),
            "validation_qwk": point(validation.get("QWK Score", "")),
            "validation_macro_f1": point(validation.get("Macro F1", validation.get("Macro Recall / F1", ""))),
            "validation_grade1_recall": point(validation.get("Grade 1 Recall", "")),
            "validation_average_precision": point(validation.get("Average Precision", validation.get("Average Precision / ROC AUC", ""))),
            "validation_roc_auc": "not reported",
            "validation_selection_score": point(validation.get("Composite Selection Score", "")),
            "best_validation_epoch": config.get("Selected Checkpoint", config.get("Selected Loss / Epoch", "not reported")),
            "notes": "Values are transcribed from the consolidated report; see source for epoch history, confusion matrix, class report, and figures.",
        })
        combined = metrics.get("Macro Precision / Recall / F1")
        if combined:
            numbers = re.findall(r"0\.\d+", combined)
            if len(numbers) >= 3:
                row["test_macro_precision"], row["test_macro_recall"], row["test_macro_f1"] = numbers[:3]
        grade1 = metrics.get("Grade 1 Precision / Recall")
        if grade1:
            numbers = re.findall(r"0\.\d+", grade1)
            if len(numbers) >= 2:
                row["test_grade1_precision"], row["test_grade1_recall"] = numbers[:2]
        val_pair = validation.get("Average Precision / ROC AUC")
        if val_pair:
            numbers = re.findall(r"0\.\d+", val_pair)
            if len(numbers) >= 2:
                row["validation_average_precision"], row["validation_roc_auc"] = numbers[:2]
        extra = top.get("extra", [])
        if model_family == "DenseNet-121" and len(extra) >= 4:
            row["validation_failures"] = extra[0]
            row["boundary_confusions"] = extra[1]
            row["critical_under_predictions"] = extra[2]
            row["critical_over_predictions"] = extra[3]
        elif model_family.startswith("SE-ResNeXt") and len(extra) >= 4:
            row["test_macro_f1"] = point(extra[0])
            row["test_grade1_recall"] = point(extra[1])
            row["cam_joint_energy"] = point(extra[2])
            row["cam_border_energy"] = point(extra[3])
        records.append(row)

    existing = {row["run_timestamp"][:19] for row in records}
    for key, top in summary.items():
        if key in existing:
            continue
        row = {field: "not reported" for field in FIELDS}
        row.update({
            "model_family": model_family, "run_timestamp": key,
            "timezone": "as shown in report summary or not recorded",
            "record_type": "summary-only run", "status": "completed; provenance incomplete",
            "short_description": top["description"],
            "purpose": top["description"], "source_report": str(report_path.relative_to(ROOT)),
            "test_accuracy": top["accuracy"], "test_qwk": top["qwk"],
            "test_roc_auc": top["auc"], "test_average_precision": top["ap"],
            "confidence_interval_notes": top["ci"],
            "notes": "No dedicated detailed run section was found; only the report summary row is indexed.",
        })
        records.append(row)
    return records


def add_ablation_rows(records: list[dict[str, str]], directory_name: str, model: str) -> None:
    report_path = REPORT_ROOT / directory_name / "report.md"
    report = report_path.read_text(encoding="utf-8")
    pattern = re.compile(r"^### Run: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?) UTC \(([^)]+)\)", re.M)
    matches = list(pattern.finditer(report))
    for index, match in enumerate(matches):
        section = report[match.start(): matches[index + 1].start() if index + 1 < len(matches) else report.find("\n## Run:", match.end())]
        tables = []
        for table in re.findall(r"(?:^\|.*\|\n){3,}", section, re.M):
            lines = [clean(line) for line in table.strip().splitlines() if "---" not in line]
            tables.append(" / ".join(lines))
        timestamp, description = match.group(1), clean(match.group(2))
        row = {field: "not reported" for field in FIELDS}
        row.update({
            "model_family": model, "run_timestamp": timestamp, "timezone": "UTC",
            "record_type": "validation/post-hoc ablation", "status": "completed",
            "short_description": description,
            "purpose": clean(section.split("\n\n", 1)[1].split("\n\n", 1)[0]) if "\n\n" in section else description,
            "source_report": str(report_path.relative_to(ROOT)),
            "notebook_archive": str(find_notebook(REPORT_ROOT / directory_name, timestamp).relative_to(ROOT)) if find_notebook(REPORT_ROOT / directory_name, timestamp) else "not archived in this model report folder",
            "experiment_results": " || ".join(tables) if tables else "see source report",
            "notes": "Ablation/audit row: metrics are validation or post-hoc comparison results, not a new locked-test checkpoint.",
        })
        if "GRAD-CAM VS NATIVE CAM" in description:
            row["heatmap_method"] = "Final-layer Grad-CAM compared with native CAM"
            row["decision"] = "No demonstrated superiority; retain native CAM for cheaper, structurally faithful inference."
            row["known_issue"] = "Checkpoint resolver selected stage2_best_model.pth by substring; exact final-checkpoint audit requires rerun."
        elif "SAMPLER" in description:
            row["compared_arms"] = "full inverse; square-root inverse; no sampler"
            row["purpose"] = "Compare class sampling strategies under the same validation objective and CAM audit."
        elif "JOINT-GUIDED" in description:
            row["compared_arms"] = "CE control; joint guidance 0.02; joint guidance 0.05"
            row["purpose"] = "Test weak broad-band joint guidance against the CE control using validation metrics and CAM proxies."
        elif "CAM ARCHITECTURE" in description:
            row["compared_arms"] = "final native CAM CE; ordinal soft label; joint guidance; multiscale HiResCAM; FPN native CAM"
            row["purpose"] = "Compare CAM heads, CE/ordinal losses, and weak localization guidance on the same validation split."
        records.append(row)


def apply_known_values(records: list[dict[str, str]]) -> None:
    values = {
        ("DenseNet-121", "2026-07-25 06:30:25"): {
            "status": "selected/deployed for external audit", "selected_epoch": "24 (fine-tune)",
            "loss_function": "Cross-Entropy (CE), selected", "compared_arms": "CE; ordinal PD-2; CE + 0.25 ordinal PD-2",
            "input_size": "384x384 direct resize after square padding",
            "pipeline": "3-stage: 5 head warm-up + 15 coarse + 10 full fine-tune",
            "cam_cases": "227", "cam_joint_energy": "0.8235", "cam_border_energy": "0.1130",
            "cam_lower_tibia_energy": "0.0884", "cam_peak_inside_rate": "0.9956",
            "cam_occlusion_metric": "joint-occlusion probability drop 0.5428", "cam_flip_correlation": "0.9609",
            "decision": "Retain CE for classification; external CAM localization is not production-grade.",
            "known_issue": "External YOLO ROI/domain shift: 144/217 CAMs failed the conservative anatomy gate.",
        },
        ("DenseNet-121", "2026-07-25 04:34:08"): {
            "decision": "Rejected; do not replace the previous checkpoint.",
            "known_issue": "EMA/natural-orientation run substantially reduced classification performance.",
        },
        ("DenseNet-121", "2026-07-21 15:07:17"): {
            "image_processing_full": "Square-pad complete ROI; LAB CLAHE; resize 400x400; train random crop or validation/test center crop to 384x384; ToTensor; ImageNet normalization",
            "training_augmentation_full": "Laterality canonicalization before transforms; horizontal flip disabled; mild rotation/brightness/contrast pipeline; Random Erasing p=0.10; minority augmentation disabled",
            "validation_test_processing": "Canonicalize right knees; square-pad; LAB CLAHE; resize 400x400; center crop 384x384; ToTensor; ImageNet normalization; no TTA",
            "heatmap_method": "Five bias-free 1x1 class maps; global spatial mean logits; positive predicted-grade native CAM",
            "heatmap_resolution": "12x12 final feature map",
        },
        ("DenseNet-121", "2026-07-23 01:31:37"): {
            "image_processing_full": "Square-pad complete ROI; LAB CLAHE; resize 400x400; train random crop or validation/test center crop to 384x384; ToTensor; ImageNet normalization",
            "training_augmentation_full": "Laterality canonicalization before transforms; horizontal flip disabled; mild rotation/brightness/contrast pipeline; Random Erasing p=0.10; minority augmentation disabled",
            "validation_test_processing": "Canonicalize right knees; square-pad; LAB CLAHE; resize 400x400; center crop 384x384; ToTensor; ImageNet normalization; no TTA",
            "heatmap_method": "Five bias-free 1x1 class maps; global spatial mean logits; positive predicted-grade native CAM",
            "heatmap_resolution": "12x12 final feature map",
        },
        ("SE-ResNeXt-50 32x4d", "2026-07-23 01:25:36"): {
            "cam_cases": "227", "cam_joint_energy": "0.8707", "cam_border_energy": "0.0749",
            "cam_lower_tibia_energy": "0.0880", "cam_peak_inside_rate": "0.9956",
            "decision": "Retained canonical 12x12 native-CAM CE baseline.",
        },
        ("SE-ResNeXt-50 32x4d", "2026-07-23 06:57:13"): {
            "cam_cases": "227", "cam_joint_energy": "0.7938", "cam_border_energy": "0.1175",
            "cam_lower_tibia_energy": "0.1149", "cam_peak_inside_rate": "0.9912",
            "decision": "Rejected; higher map resolution and Grade 1 recall did not offset metric/CAM regressions.",
            "known_issue": "EMA lag and multiscale-head change were confounded in one experiment.",
        },
        ("SE-ResNeXt-50 32x4d", "2026-07-25 01:50:53"): {
            "cam_cases": "227", "cam_joint_energy": "0.8516", "cam_border_energy": "0.0990",
            "cam_lower_tibia_energy": "0.0878", "cam_peak_inside_rate": "0.9912",
            "decision": "Preferred SE-ResNeXt candidate for single-knee inputs without laterality metadata; external comparison still required.",
        },
    }
    for row in records:
        override = values.get((row["model_family"], row["run_timestamp"][:19]))
        if override:
            row.update(override)
        if row["model_family"] == "DenseNet-121" and row["run_timestamp"][:19] in {
            "2026-07-17 22:15:13", "2026-07-18 20:27:46", "2026-07-18 22:03:35",
        }:
            row["status"] = "invalid implementation; historical result only"
            row["known_issue"] = "Documented logic error: intended backbone blocks remained frozen."
            row["decision"] = "Do not use for model selection or production."
        timestamp = row["run_timestamp"][:19]
        if row["model_family"] == "DenseNet-121" and timestamp <= "2026-07-18 22:03:35" and row["record_type"] == "training run":
            row["image_processing_full"] = (
                f"Square-pad complete ROI; LAB CLAHE (OpenCV implementation); resize to {row['input_size']}; "
                "ToTensor; ImageNet mean/std normalization"
            )
            if timestamp == "2026-07-15 13:42:33":
                row["training_augmentation_full"] = "Horizontal flip p=0.50; rotation +/-8 degrees; Random Erasing disabled; no minority augmentation"
            else:
                row["training_augmentation_full"] = "Horizontal flip p=0.50; rotation +/-8 degrees; double Random Erasing p=0.80 for regular train and p=0.90 for minority transform, scale 0.02-0.15, ratio 0.3-3.3; minority augmentation enabled"
            row["validation_test_processing"] = f"Square-pad; LAB CLAHE; resize to {row['input_size']}; ToTensor; ImageNet normalization; no stochastic augmentation"
        if row["model_family"] == "DenseNet-121" and timestamp == "2026-07-20 12:36:36":
            row["image_processing_full"] = "Square-pad complete ROI; LAB CLAHE; resize 400x400; crop 384x384; ToTensor; ImageNet normalization"
            row["training_augmentation_full"] = "Horizontal flip p=0.50; rotation +/-8 degrees; random crop 384x384; one Random Erasing p=0.10, scale 0.02-0.05, ratio 0.5-2.0; minority augmentation disabled"
            row["validation_test_processing"] = "Square-pad; LAB CLAHE; resize 400x400; center crop 384x384; ToTensor; ImageNet normalization; no TTA"
        if row["model_family"].startswith("SE-ResNeXt") and row["record_type"] == "training run":
            row["image_processing_full"] = "Square-pad complete ROI; LAB CLAHE; resize 400x400; crop 384x384; ToTensor; ImageNet normalization"
            row["validation_test_processing"] = "Apply matching laterality policy; square-pad; LAB CLAHE; resize 400x400; center crop 384x384; ToTensor; ImageNet normalization; no stochastic augmentation"


def add_external_dense_audit(records: list[dict[str, str]]) -> None:
    summary_path = REPORT_ROOT / "dense_net_121" / "assets" / "2026-07-25_15-32-33_UTC_api_cam_localization_audit" / "summary.json"
    report_path = REPORT_ROOT / "dense_net_121" / "2026-07-25_15-32-33_UTC_api_cam_localization_audit.md"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    row = {field: "not reported" for field in FIELDS}
    row.update({
        "model_family": "DenseNet-121", "run_timestamp": "2026-07-25 15:32:33",
        "timezone": "UTC", "record_type": "external API CAM audit",
        "status": "completed; localization gate failed for majority of CAMs",
        "short_description": "Full production API native-CAM localization audit",
        "purpose": "Submit all 105 unlabeled external test_images through YOLO and DenseNet, verify the response schema, and quantify native-CAM anatomy-gate behavior.",
        "decision": "API/schema accepted; explanations not validated as production-grade anatomical localization.",
        "known_issue": "138/209 CAMs failed: upper femur/top border, far lateral border, lower tibia/bottom border, or diffuse/off-joint activation.",
        "source_report": str(report_path.relative_to(ROOT)),
        "architecture": "Deployed DenseNet-121 final-linear native-CAM checkpoint behind YOLO ROI detection",
        "heatmap_method": "Predicted-grade native CAM",
        "dataset_split": "105 unlabeled external application images; no KL accuracy metrics can be computed",
        "test_n": str(payload["knees"]), "cam_cases": str(payload["knees"]),
        "cam_notes": (
            f"passed={payload['passed_cams']}; failed={payload['failed_cams']}; "
            f"unique_failed_images={payload['unique_failed_images']}; reasons={payload['reason_counts']}; "
            "passing mean joint/border/lower-tibia=0.819/0.111/0.070; "
            "failing mean joint/border/lower-tibia=0.291/0.312/0.206"
        ),
        "experiment_results": f"prediction_grade_counts={payload['grade_counts']}; failure_patterns={payload['pattern_counts']}; api_schema_unchanged={payload['api_schema_unchanged']}; mean_request_seconds={payload['mean_request_seconds']:.3f}; max_request_seconds={payload['max_request_seconds']:.3f}",
        "notes": "This is an unlabeled operational/localization audit, not a training run and not an accuracy evaluation.",
    })
    records.append(row)


def write_csv(path: Path, records: list[dict[str, str]]) -> None:
    records.sort(key=lambda row: (row["run_timestamp"], row["record_type"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    dense = parse_model("dense_net_121", "DenseNet-121")
    seresnext = parse_model("se_resnext50_32x4d", "SE-ResNeXt-50 32x4d")
    add_ablation_rows(dense, "dense_net_121", "DenseNet-121")
    add_ablation_rows(seresnext, "se_resnext50_32x4d", "SE-ResNeXt-50 32x4d")
    apply_known_values(dense)
    apply_known_values(seresnext)
    add_external_dense_audit(dense)
    write_csv(REPORT_ROOT / "dense_net_121" / "experiment_summary.csv", dense)
    write_csv(REPORT_ROOT / "se_resnext50_32x4d" / "experiment_summary.csv", seresnext)
    write_csv(REPORT_ROOT / "model_experiment_summary.csv", dense + seresnext)
    print(f"DenseNet rows: {len(dense)}")
    print(f"SE-ResNeXt rows: {len(seresnext)}")


if __name__ == "__main__":
    main()
