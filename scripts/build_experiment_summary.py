#!/usr/bin/env python3
"""Build a read-only experiment inventory from the repository's report artifacts.

The repository intentionally keeps historical experiment records in CSV/Markdown
reports. This utility joins those records into one workbook and exports each tab
as a CSV, without opening or changing model checkpoints.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "report"
OUT = REPORT / "summary"
XLSX = REPORT / "all_experiments.xlsx"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(value: object) -> str:
    """Return a stable decimal string for workbook numeric cells."""
    if value is None or value == "":
        return ""
    text = str(value).strip()
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return match.group(0) if match else ""


def base_row(row: dict[str, str], source: str) -> dict[str, str]:
    result = dict(row)
    result["source_file"] = source
    result["record_origin"] = "report_csv"
    result.setdefault("checkpoint_path", "")
    result.setdefault("checkpoint_sha256", "")
    result.setdefault("checkpoint_epoch", "")
    return result


def manual_row(**values: object) -> dict[str, str]:
    row = {str(key): "" if value is None else str(value) for key, value in values.items()}
    row.setdefault("source_file", "repository artifact")
    row.setdefault("record_origin", "manual_artifact_metadata")
    return row


def collect_rows() -> list[dict[str, str]]:
    sources = [
        (REPORT / "dense_net_121" / "experiment_summary.csv", 0),
        (REPORT / "se_resnext50_32x4d" / "experiment_summary.csv", 0),
        (REPORT / "model_experiment_summary.csv", 1),
    ]
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    # Model-specific summaries are preferred over the older global copy.
    for path, _priority in sources:
        for row in read_csv(path):
            item = base_row(row, str(path.relative_to(ROOT)))
            key = (item.get("model_family", ""), item.get("run_timestamp", ""), item.get("short_description", ""))
            if key in seen:
                continue
            seen.add(key)
            rows.append(item)

    cutout_root = ROOT / "2026-07-28_02-00-33_963495_UTC_cutout_ablation"
    cutout_csv = cutout_root / "arm_comparison.csv"
    if cutout_csv.exists():
        cutout_hashes = {
            "aggressive_single_cutout": "f955d0f516cd30d3c057c822e97219841e24da74dabbbc9f5dad134a31b43dd6",
            "aggressive_double_cutout": "aa774a2fc765244e03d29e622e31ce0f8fa33a27aeaae8fbf5c5ec312f447fc2",
            "baseline_random_erasing": "0c26982e344a339c247d4fc7d3a65672cade9c912d9f7f51b7f1a3256c430b28",
        }
        for arm in read_csv(cutout_csv):
            name = arm["arm"]
            checkpoint = cutout_root / name / "best_model.pth"
            rows.append(manual_row(
                model_family="DenseNet-121",
                run_timestamp="2026-07-28 02:00:33.963495",
                timezone="UTC",
                record_type="validation/CAM ablation",
                status="completed; not promoted",
                short_description=f"Cutout ablation: {name}",
                purpose="Test stronger masking against shortcut learning with the same CE DenseNet pipeline.",
                decision="Retain current production checkpoint; no arm improved the complete selection trade-off.",
                known_issue="No labeled test evaluation in this run; external YOLO-crop shift remains unresolved.",
                source_report=str((cutout_root / "report.md").relative_to(ROOT)),
                notebook_archive="not archived in report folder",
                checkpoint_directory=str(checkpoint.relative_to(ROOT)),
                checkpoint_path=str(checkpoint.relative_to(ROOT)),
                checkpoint_sha256=cutout_hashes.get(name, sha256(checkpoint) if checkpoint.exists() else ""),
                checkpoint_epoch=arm["best_epoch"],
                architecture="timm_densenet121_linear_native_cam",
                initialization="ImageNet pretrained",
                input_size="384x384 direct resize after square pad",
                roi_and_crop_policy="Complete square-padded knee ROI; no center crop",
                image_processing_full="LAB CLAHE clipLimit 1.25; SquarePad; direct Resize 384; ImageNet normalization",
                training_augmentation_full=(
                    "Natural orientation; horizontal flip p=0.50; rotation +/-5 degrees; "
                    "brightness/contrast 0.08; RandomErasing baseline p=0.10"
                ),
                validation_test_processing="Deterministic square-pad/CLAHE/resize; no stochastic augmentation",
                laterality_policy="Natural left/right orientation; no deterministic mirroring",
                sampler="Full inverse-frequency WeightedRandomSampler",
                pipeline="3-stage: 5 warm-up + 15 coarse + 10 fine-tune",
                training_stages="5 head warm-up; 15 coarse; 10 full fine-tune",
                epochs="30",
                selected_epoch=arm["best_epoch"],
                batch_size="48",
                workers_gpu="4 / Tesla T4",
                loss_function="Cross-Entropy (CE)",
                optimizer="AdamW",
                scheduler="CosineAnnealingLR",
                learning_rates="warm-up 3e-4; coarse backbone 3e-5/head 3e-4; fine-tune 1e-5",
                weight_decay="1e-4",
                amp="CUDA AMP",
                gradient_clipping="global norm 1.0",
                ema="disabled",
                seed="42",
                checkpoint_selection="0.55 QWK + 0.30 macro F1 + 0.15 macro AP",
                heatmap_method="Positive predicted-grade native CAM from five 1x1 class maps",
                heatmap_resolution="12x12",
                dataset_split="5,778 train / 826 validation; test not read",
                validation_qwk=arm["qwk"],
                validation_macro_f1=arm["macro_f1"],
                validation_grade1_recall=arm["grade1_recall"],
                validation_average_precision=arm["macro_ap"],
                validation_selection_score=arm["selection_score"],
                best_validation_epoch=arm["best_epoch"],
                cam_cases="826",
                cam_notes=f"Anatomy-gate failure rate {float(arm['cam_failure_rate']):.4%}; see per-case CSV in run folder.",
                validation_failures=arm["cam_failure_rate"],
                experiment_results=json.dumps(arm, sort_keys=True),
                notes="Baseline random erasing had the best Grade 1 recall and macro F1; aggressive single cutout had the best QWK/AP but was not a production replacement.",
            ))

    preprocessing_arms = [
        ("raw_then_pad", "29", "0.6489", "0.8117", "0.6810", "0.4444", "0.7305", "0.8899", "0.7603", "0.8310", "0.1111"),
        ("pad_then_clahe2_current", "30", "0.6489", "0.8110", "0.6867", "0.4314", "0.7317", "0.8881", "0.7618", "0.8237", "0.1158"),
        ("clahe2_then_pad", "27", "0.6598", "0.8176", "0.6879", "0.4248", "0.7341", "0.8896", "0.7661", "0.8243", "0.1171"),
        ("clahe1_25_then_pad", "30", "0.6695", "0.8274", "0.7061", "0.5294", "0.7411", "0.8951", "0.7781", "0.8339", "0.1092"),
        ("percentile_1_99_then_pad", "27", "0.6671", "0.8242", "0.6797", "0.4444", "0.7310", "0.8964", "0.7669", "0.8283", "0.1124"),
        ("clahe1_25_pad_acquisition_robustness", "30", "0.6477", "0.8082", "0.6930", "0.4837", "0.7458", "0.8941", "0.7643", "0.8360", "0.1115"),
    ]
    for name, epoch, acc, qwk, f1, g1, ap, auc, selection, joint, border in preprocessing_arms:
        rows.append(manual_row(
            model_family="DenseNet-121", run_timestamp="2026-07-25 23:48:22.997435", timezone="UTC",
            record_type="validation preprocessing ablation", status="completed; confirmation required",
            short_description=f"Preprocessing quality arm: {name}",
            purpose="Test acquisition preprocessing order and strength under the same DenseNet/CE training protocol.",
            decision="CLAHE 1.25 -> pad is the validation winner; do not promote until a clean confirmation and locked holdout.",
            known_issue="Validation-only run; no test or labeled production-YOLO evaluation.",
            source_report="docs/report/dense_net_121/2026-07-25_23-48-22_preprocessing_quality_ablation.md",
            notebook_archive="docs/report/dense_net_121/2026-07-25_23-48-22_densenet121_preprocessing_quality_ablation.ipynb",
            architecture="timm_densenet121_linear_native_cam", initialization="ImageNet pretrained",
            input_size="384x384 direct resize after square pad", roi_and_crop_policy="Complete ROI; no center crop",
            image_processing_full=("Raw -> pad" if name == "raw_then_pad" else "Pad -> CLAHE 2.0" if name == "pad_then_clahe2_current" else "CLAHE 2.0 -> pad" if name == "clahe2_then_pad" else "CLAHE 1.25 -> pad" if name == "clahe1_25_then_pad" else "Percentile 1-99 -> pad" if name == "percentile_1_99_then_pad" else "CLAHE 1.25 -> pad + gamma/blur/sharpness/noise"),
            training_augmentation_full="Horizontal flip p=0.50; rotation +/-5; brightness/contrast 0.08; RandomErasing p=0.10",
            validation_test_processing="Deterministic arm-specific preprocessing; no test opened",
            laterality_policy="Natural left/right orientation; no deterministic mirroring", sampler="Full inverse-frequency",
            pipeline="5 head warm-up + 15 coarse + 10 fine-tune", training_stages="5/15/10", epochs="30", selected_epoch=epoch,
            batch_size="48", workers_gpu="Tesla T4", loss_function="Cross-Entropy (CE)", optimizer="AdamW",
            scheduler="CosineAnnealingLR", learning_rates="warm-up 3e-4; coarse 3e-5/3e-4; fine 1e-5", weight_decay="1e-4",
            amp="CUDA AMP", gradient_clipping="global norm 1.0", ema="disabled", seed="42",
            checkpoint_selection="0.55 QWK + 0.30 macro F1 + 0.15 macro AP", heatmap_method="Positive predicted-grade native CAM",
            heatmap_resolution="12x12", dataset_split="5,778 train / 826 validation; test not read",
            validation_accuracy=acc, validation_qwk=qwk, validation_macro_f1=f1, validation_grade1_recall=g1,
            validation_average_precision=ap, validation_roc_auc=auc, validation_selection_score=selection,
            best_validation_epoch=epoch, cam_cases="227", cam_joint_energy=joint, cam_border_energy=border,
            cam_notes="Selected arm audited 225/227 CAMs through the broad anatomy gate; remaining arms have aggregate gate figures in the source report.",
            source_file="docs/report/dense_net_121/2026-07-25_23-48-22_preprocessing_quality_ablation.md",
            record_origin="report_markdown",
        ))

    rows.extend([
        manual_row(
            model_family="DenseNet-201",
            run_timestamp="not recorded",
            timezone="not recorded",
            record_type="training/evaluation report",
            status="completed; separate model report",
            short_description="Optimized multi-scale DenseNet-201",
            purpose="Evaluate a larger multi-scale DenseNet backbone with ordinal training and Grad-CAM.",
            decision="Retain as a documented candidate; QWK did not beat the best DenseNet-121 runs and Grade 1 recall is weak.",
            known_issue="Report does not provide an exact UTC run timestamp or checkpoint provenance; do not use as an exact production record.",
            source_report="docs/report/dense_net_201/dense_net_201_test_results.md",
            notebook_archive="dense_net_201.ipynb (path not present in report folder)",
            checkpoint_directory="checkpoints/densenet201",
            checkpoint_path="checkpoints/densenet201/best_model.pth",
            checkpoint_sha256="a8107b9cc7cc9242385f1facfcfc69c251f88697ed2df4dae8a43c1d66729b76",
            architecture="Multi-scale DenseNet-201; pooled transition-2, transition-3, and norm5 features",
            initialization="ImageNet pretrained",
            input_size="300x300",
            roi_and_crop_policy="Square-padded knee ROI; exact crop provenance not documented",
            image_processing_full="Square padding; CLAHE; grayscale normalization; resize as configured",
            training_augmentation_full="Horizontal flip p=0.50; capped rotation; small RandomErasing; no color jitter",
            validation_test_processing="Deterministic preprocessing; exact sequence not fully recorded",
            laterality_policy="Natural/symmetric laterality; exact policy not recorded",
            sampler="Balanced WeightedRandomSampler in warm-up/coarse stages; stage-3 natural skew",
            pipeline="3-stage warm-up + coarse + fine-tune",
            training_stages="5 head warm-up; 25 coarse; 15 focal-CORN fine-tune (per configuration guide)",
            epochs="45",
            batch_size="not recorded",
            workers_gpu="not recorded",
            loss_function="CORN; Focal CORN in final stage",
            optimizer="AdamW",
            scheduler="Cosine annealing; eta_min 1e-7",
            learning_rates="backbone 1e-5; head 1e-4; final 1e-5",
            weight_decay="1e-4 stages 1/2; 1e-3 stage 3",
            amp="not recorded",
            ema="not recorded",
            seed="not recorded",
            checkpoint_selection="not recorded",
            heatmap_method="Grad-CAM over norm5",
            heatmap_resolution="not recorded",
            dataset_split="1,656-image test split; patient grouping not documented",
            test_n="1656",
            test_accuracy="0.6624",
            test_qwk="0.7763",
            test_macro_f1="0.63",
            test_average_precision="0.6948",
            test_roc_auc="0.8812",
            validation_failures="288 / 826",
            notes="DenseNet-201 report recommends balanced stage-3 training and threshold review, but those recommendations were not validated in the documented run.",
            source_file="docs/report/dense_net_201/dense_net_201_test_results.md",
            record_origin="report_markdown",
        ),
        manual_row(
            model_family="EfficientNet-B0",
            run_timestamp="2026-07-24 04:45:25.604705",
            timezone="UTC", record_type="training run", status="completed; not promoted",
            short_description="EfficientNet-B0 final 12x12 native-CAM CE",
            purpose="Standalone EfficientNet scale candidate and CAM audit.",
            decision="Do not promote; weaker classifier than DenseNet/SE-ResNeXt.",
            source_report="docs/report/efficientnet_b0/report.md",
            notebook_archive="docs/report/efficientnet_b0/2026-07-24_04-45-25_efficientnet_b0_final_native_cam_ce.ipynb",
            checkpoint_directory="checkpoints/efficientnet_b0", checkpoint_path="checkpoints/efficientnet_b0/best_model.pth",
            checkpoint_sha256="47238a3ee5350b6521e3f292d30493e7a7e37d0c7aee748b46424c0859fe60ff",
            architecture="efficientnet_b0_final_native_cam_ce", initialization="ImageNet pretrained",
            input_size="384x384 crop from 400x400", roi_and_crop_policy="Square pad; crop 384 from resized ROI",
            image_processing_full="SquarePad; LAB CLAHE; Resize 400; Crop 384; ToTensor; ImageNet normalization",
            training_augmentation_full="Canonical right-knee mirroring; rotation; brightness/contrast; RandomErasing",
            validation_test_processing="Canonical mirror; square pad; CLAHE; resize/crop; deterministic",
            laterality_policy="Right knees mirrored to canonical orientation", sampler="Full inverse-frequency",
            pipeline="5 warm-up + 15 coarse + 10 fine-tune", training_stages="5/15/10", epochs="30", selected_epoch="10",
            batch_size="24, accumulation 2 (effective 48)", workers_gpu="Tesla T4", loss_function="Cross-Entropy (CE)",
            optimizer="AdamW", scheduler="CosineAnnealingLR", learning_rates="warm-up 3e-4; coarse 3e-4/3e-5; fine 1e-5",
            weight_decay="1e-4", amp="enabled", ema="disabled", seed="42",
            checkpoint_selection="validation composite", heatmap_method="12x12 native CAM", heatmap_resolution="12x12",
            dataset_split="5,778 train / 826 validation / 1,656 test", test_n="1656", test_accuracy="0.6051", test_qwk="0.7992",
            test_macro_f1="0.6258", test_grade1_recall="0.3986", test_average_precision="0.6817", test_roc_auc="0.8723",
            validation_accuracy="0.6150", validation_qwk="0.7743", validation_macro_f1="0.6317", validation_grade1_recall="0.4771",
            validation_average_precision="0.6690", validation_roc_auc="0.8660", validation_selection_score="0.6936",
            best_validation_epoch="10", cam_cases="227", cam_joint_energy="0.8280", cam_border_energy="0.1080",
            cam_lower_tibia_energy="0.0797", cam_peak_inside_rate="0.9956", cam_occlusion_metric="0.6172",
            cam_notes="Broadly joint-localized but with boundary hotspots; no standalone promotion.",
            source_file="docs/report/efficientnet_b0/report.md", record_origin="report_markdown",
        ),
        manual_row(
            model_family="YOLOv8",
            run_timestamp="2026-07-26 07:25:09",
            timezone="UTC", record_type="detector training", status="current checkpoint",
            short_description="YOLOv8n joint detector current best.pt", purpose="Detect knee joint ROI in full radiographs.",
            decision="Current detector; use with classifier crop pipeline.", source_report="checkpoints/yolov8/best.pt",
            checkpoint_directory="checkpoints/yolov8", checkpoint_path="checkpoints/yolov8/best.pt",
            checkpoint_sha256="443003a317ca71f3f63e0f5ac145a0eafa431053a998dbb4c60c65b5c5e7fef7",
            architecture="YOLOv8n", input_size="640", image_processing_full="Ultralytics YOLO preprocessing",
            training_augmentation_full="Ultralytics default detection augmentation (train args in checkpoint metadata)",
            epochs="100", batch_size="16", workers_gpu="YOLO training runtime", loss_function="YOLO box/cls/DFL losses",
            validation_precision="0.99952", validation_recall="0.99145", validation_map50="0.99500", validation_map50_95="0.90477",
            notes="Detector metrics are not KL-grading metrics; downstream CAM can still fail when classifier training crops differ from YOLO crops.",
            source_file="checkpoints/yolov8/best.pt", record_origin="checkpoint_metadata",
        ),
        manual_row(
            model_family="YOLOv8", run_timestamp="2026-07-14 16:03:31.090356", timezone="UTC",
            record_type="detector training", status="archived",
            short_description="YOLOv8 archived detector", purpose="Previous knee ROI detector for comparison.",
            decision="Retired; lower mAP50-95 than current checkpoint.", source_report="checkpoints/yolov8/archive/best.pt",
            checkpoint_directory="checkpoints/yolov8/archive", checkpoint_path="checkpoints/yolov8/archive/best.pt",
            checkpoint_sha256="3e18a09e58df0ae1f8e3102d5e63893c72d2508bba5ffd80c84dac8356161d2b",
            architecture="YOLOv8n", validation_precision="0.98797", validation_recall="0.98812", validation_map50="0.98871",
            validation_map50_95="0.81358", source_file="checkpoints/yolov8/archive/best.pt", record_origin="checkpoint_metadata",
        ),
        manual_row(
            model_family="Ensemble", run_timestamp="2026-07-24 09:03:08.287732", timezone="UTC",
            record_type="deployment smoke test", status="experimental; unlabeled",
            short_description="Three-model weighted soft vote (DenseNet + SE-ResNeXt + EfficientNet-B0)",
            purpose="Probability-level ensemble and dynamic heatmap source selection.",
            decision="Operational only; do not claim accuracy from unlabeled smoke images.",
            source_report="docs/report/ensemble/2026-07-24_09-03-08_three_model_weighted_soft_voting.md",
            architecture="0.50 DenseNet / 0.35 SE-ResNeXt / 0.15 EfficientNet-B0",
            input_size="384x384", roi_and_crop_policy="YOLO joint ROIs", image_processing_full="Per-model checkpoint preprocessing",
            pipeline="Soft-vote probabilities; choose agreeing CAM by probability x joint reliability",
            heatmap_method="Dynamic native-CAM component selection", dataset_split="20 unlabeled images; 40 knee ROIs",
            notes="19 automated checks passed; 20/20 HTTP 200; response schema unchanged; not an accuracy evaluation.",
            source_file="docs/report/ensemble/2026-07-24_09-03-08_three_model_weighted_soft_voting.md", record_origin="report_markdown",
        ),
    ])
    return rows


def artifact_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    known = {Path(r.get("notebook_archive", "")).name for r in rows if r.get("notebook_archive")}
    result = []
    for notebook in sorted(REPORT.rglob("*.ipynb")):
        result.append({
            "artifact": str(notebook.relative_to(ROOT)),
            "artifact_type": "notebook",
            "status": "matched to completed row" if notebook.name in known else "archived/unmatched; inspect before citing",
            "run_timestamp": (re.match(r"(\d{4}-\d{2}-\d{2}[T_\-\d]*)", notebook.name) or [""])[0],
        })
    for doc in sorted(REPORT.rglob("*.md")):
        result.append({"artifact": str(doc.relative_to(ROOT)), "artifact_type": "report markdown", "status": "source report", "run_timestamp": ""})
    planned = (
        ROOT
        / "notebooks"
        / "experiments"
        / "densenet121"
        / "heatmaps"
        / "dense_net_121_roi_robustness_ablation.ipynb"
    )
    if planned.exists():
        result.append({"artifact": str(planned.relative_to(ROOT)), "artifact_type": "notebook", "status": "planned/not run", "run_timestamp": ""})
    recent_run = ROOT / "2026-07-28_02-00-33_963495_UTC_cutout_ablation"
    if recent_run.exists():
        for artifact in (recent_run / "report.md", recent_run / "arm_comparison.csv", recent_run / "gradcam_comparison_summary.json"):
            if artifact.exists():
                result.append({
                    "artifact": str(artifact.relative_to(ROOT)),
                    "artifact_type": artifact.suffix.lstrip(".") or "run artifact",
                    "status": "completed run artifact",
                    "run_timestamp": "2026-07-28 02:00:33.963495 UTC",
                })
    return result


def checkpoint_rows() -> list[dict[str, str]]:
    paths = [
        ("DenseNet-121", ROOT / "checkpoints/densenet121/best_model.pth", "current app checkpoint"),
        ("DenseNet-201", ROOT / "checkpoints/densenet201/best_model.pth", "available checkpoint; not current app mode"),
        ("EfficientNet-B0", ROOT / "checkpoints/efficientnet_b0/best_model.pth", "standalone candidate"),
        ("SE-ResNeXt-50 32x4d", ROOT / "checkpoints/se_resnext50_32x4d/best_model (1).pth", "available checkpoint; filename should be pinned in deployment metadata"),
        ("YOLOv8", ROOT / "checkpoints/yolov8/best.pt", "current detector"),
        ("YOLOv8", ROOT / "checkpoints/yolov8/archive/best.pt", "archived detector"),
    ]
    result = []
    for model, path, note in paths:
        result.append({
            "model_family": model, "path": str(path.relative_to(ROOT)), "exists": str(path.exists()).lower(),
            "sha256": sha256(path) if path.exists() else "", "size_bytes": str(path.stat().st_size) if path.exists() else "",
            "role": note,
        })
    return result


def metrics_long(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    fields = [
        ("test", "test_accuracy"), ("test", "test_qwk"), ("test", "test_macro_f1"),
        ("test", "test_grade1_recall"), ("test", "test_average_precision"), ("test", "test_roc_auc"),
        ("validation", "validation_accuracy"), ("validation", "validation_qwk"),
        ("validation", "validation_macro_f1"), ("validation", "validation_grade1_recall"),
        ("validation", "validation_average_precision"), ("validation", "validation_roc_auc"),
        ("validation", "validation_selection_score"), ("cam", "cam_joint_energy"),
        ("cam", "cam_border_energy"), ("cam", "cam_lower_tibia_energy"),
        ("cam", "cam_peak_inside_rate"), ("cam", "cam_occlusion_metric"),
        ("detector_validation", "validation_precision"), ("detector_validation", "validation_recall"),
        ("detector_validation", "validation_map50"), ("detector_validation", "validation_map50_95"),
    ]
    output = []
    for row in rows:
        for split, field in fields:
            value = number(row.get(field, ""))
            if value:
                output.append({
                    "model_family": row.get("model_family", ""), "run_timestamp": row.get("run_timestamp", ""),
                    "short_description": row.get("short_description", ""), "split": split,
                    "metric": field.removeprefix(f"{split}_"), "value": value,
                    "source_file": row.get("source_file", ""),
                })
    return output


def xlsx_column_name(index: int) -> str:
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def make_xlsx(sheets: dict[str, list[dict[str, str]]]) -> None:
    XLSX.parent.mkdir(parents=True, exist_ok=True)
    names = list(sheets)
    workbook = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
        "<sheets>",
    ]
    for i, name in enumerate(names, 1):
        workbook.append(f'<sheet name="{escape(name[:31])}" sheetId="{i}" r:id="rId{i}"/>')
    workbook.extend(["</sheets>", "</workbook>"])
    relationships = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
    for i in range(1, len(names) + 1):
        relationships.append(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>')
    relationships.append('</Relationships>')
    content = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">', '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>', '<Default Extension="xml" ContentType="application/xml"/>', '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>']
    for i in range(1, len(names) + 1):
        content.append(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
    content.append('</Types>')
    with zipfile.ZipFile(XLSX, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "".join(content))
        archive.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        archive.writestr("xl/workbook.xml", "".join(workbook))
        archive.writestr("xl/_rels/workbook.xml.rels", "".join(relationships))
        for i, rows in enumerate(sheets.values(), 1):
            columns = list(rows[0]) if rows else []
            values = [columns] + [[row.get(column, "") for column in columns] for row in rows]
            sheet = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>']
            for row_number, values_row in enumerate(values, 1):
                sheet.append(f'<row r="{row_number}">')
                for col_number, value in enumerate(values_row, 1):
                    cell_ref = f"{xlsx_column_name(col_number)}{row_number}"
                    text_value = "" if value is None else str(value)
                    if row_number > 1 and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text_value.strip()):
                        sheet.append(f'<c r="{cell_ref}"><v>{text_value.strip()}</v></c>')
                    else:
                        sheet.append(f'<c r="{cell_ref}" t="inlineStr"><is><t>{escape(text_value)}</t></is></c>')
                sheet.append("</row>")
            sheet.append("</sheetData></worksheet>")
            archive.writestr(f"xl/worksheets/sheet{i}.xml", "".join(sheet))


def write_csv(name: str, rows: list[dict[str, str]]) -> None:
    path = OUT / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = collect_rows()
    rows.sort(key=lambda row: (row.get("run_timestamp", ""), row.get("model_family", ""), row.get("short_description", "")))
    densenet = [r for r in rows if "DenseNet" in r.get("model_family", "")]
    se = [r for r in rows if "SE-ResNeXt" in r.get("model_family", "")]
    efficient = [r for r in rows if "EfficientNet" in r.get("model_family", "")]
    deployment = [r for r in rows if r.get("model_family") in {"Ensemble", "YOLOv8"}]
    overview_fields = ["model_family", "run_timestamp", "status", "short_description", "purpose", "decision", "loss_function", "sampler", "pipeline", "training_augmentation_full", "heatmap_method", "test_accuracy", "test_qwk", "test_macro_f1", "test_grade1_recall", "test_average_precision", "validation_qwk", "validation_macro_f1", "validation_grade1_recall", "validation_average_precision", "cam_joint_energy", "cam_border_energy", "cam_peak_inside_rate", "validation_failures", "checkpoint_path", "checkpoint_sha256", "source_file", "notes"]
    overview = [{field: row.get(field, "") for field in overview_fields} for row in rows]
    planned = artifact_rows(rows)
    checkpoints = checkpoint_rows()
    long_metrics = metrics_long(rows)
    readme = [{
        "item": "generated_at_utc", "value": datetime.now(timezone.utc).isoformat(),
        "notes": "Generated from existing report CSV/Markdown/checkpoint artifacts; no training was started.",
    }, {
        "item": "row_counts", "value": f"all_runs={len(rows)}; DenseNet={len(densenet)}; SE-ResNeXt={len(se)}; EfficientNet={len(efficient)}; deployment={len(deployment)}",
        "notes": "Rows are historical records and recent completed cutout arms.",
    }, {
        "item": "metric_warning", "value": "Historical test metrics are retained for provenance only.",
        "notes": "The repeatedly evaluated test set must not be used to choose new configurations; future claims require a locked patient-level holdout.",
    }, {
        "item": "cam_warning", "value": "CAM numbers are not proof of clinical causality.",
        "notes": "Review per-case maps and use joint/border/occlusion audits; unlabeled API smoke tests cannot measure accuracy.",
    }, {
        "item": "source_priority", "value": "model-specific experiment_summary.csv before global model_experiment_summary.csv",
        "notes": "Duplicate rows are de-duplicated by model, timestamp, and short description.",
    }]
    sheets = {
        "README": readme,
        "overview": overview,
        "all_runs": rows,
        "densenet121": densenet,
        "se_resnext": se,
        "efficientnet": efficient,
        "deployment": deployment,
        "metrics_long": long_metrics,
        "checkpoints": checkpoints,
        "artifacts": planned,
    }
    for name, sheet_rows in sheets.items():
        write_csv(name, sheet_rows)
    make_xlsx(sheets)
    print(f"Wrote {XLSX}")
    print(f"Wrote CSV tabs to {OUT}")
    print(f"all_runs={len(rows)}; artifacts={len(planned)}; checkpoints={len(checkpoints)}; metrics={len(long_metrics)}")


if __name__ == "__main__":
    main()
