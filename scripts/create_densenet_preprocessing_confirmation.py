"""Create the focused two-arm DenseNet preprocessing confirmation notebook."""

from __future__ import annotations

import ast
import json
import re
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks/experiments/densenet121/preprocessing/dense_net_121_preprocessing_quality_ablation.ipynb"
TARGET = ROOT / "notebooks/experiments/densenet121/preprocessing/dense_net_121_preprocessing_confirmation.ipynb"

SPECS = '''DETERMINISTIC_SPECS = [
    {
        "name": "current_pad_then_clahe2",
        "enhancement": "clahe",
        "clip_limit": 2.0,
        "enhance_before_padding": False,
        "acquisition_augmentation": False,
        "loss_type": "ce",
        "ordinal_weight": 0.0,
    },
    {
        "name": "clahe1_25_then_pad",
        "enhancement": "clahe",
        "clip_limit": 1.25,
        "enhance_before_padding": True,
        "acquisition_augmentation": False,
        "loss_type": "ce",
        "ordinal_weight": 0.0,
    },
]'''

RUN_BLOCK = '''RUN_PARENT = Path("/content/drive/MyDrive/Models/densenet121_checkpoints")
requested_resume_dir = os.environ.get("PREPROCESSING_CONFIRMATION_RESUME_DIR")
incomplete_runs = sorted(
    path for path in RUN_PARENT.glob("*_preprocessing_confirmation")
    if path.is_dir() and not (path / "run_manifest.json").exists()
)
if requested_resume_dir:
    RUN_DIR = Path(requested_resume_dir)
elif incomplete_runs:
    RUN_DIR = incomplete_runs[-1]
else:
    RUN_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%f_UTC")
    RUN_DIR = RUN_PARENT / f"{RUN_TIMESTAMP}_preprocessing_confirmation"
RUN_TIMESTAMP = RUN_DIR.name.split("_preprocessing_confirmation")[0]
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"Run directory: {RUN_DIR}")'''


def replace_code(notebook: dict) -> None:
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if "RUN_PARENT = Path(" in source and "DATASET_ROOT" in source:
            source = re.sub(
                r'RUN_PARENT = Path\("/content/drive/MyDrive/Models/densenet121_checkpoints"\).*?print\(f"Run directory: \{RUN_DIR\}"\)',
                RUN_BLOCK,
                source,
                flags=re.S,
            )
        if "DETERMINISTIC_SPECS = [" in source:
            source = re.sub(
                r"DETERMINISTIC_SPECS = \[.*?\n\]\n\n\ndef build_enhancement",
                SPECS + "\n\n\ndef build_enhancement",
                source,
                flags=re.S,
            )
        if "robust_spec = dict(" in source:
            source = re.sub(
                r'deterministic_winner = choose_candidate\(arm_results\).*?selected_arm_name = choose_candidate\(arm_results\)',
                'selected_arm_name = choose_candidate(arm_results)\n'
                'deterministic_winner = selected_arm_name',
                source,
                flags=re.S,
            )
            source = source.replace(
                'print(f"Deterministic winner: {deterministic_winner}")',
                'print(f"Confirmation winner: {selected_arm_name}")',
            )
            source = source.replace(
                'print(f"Final validation-selected preprocessing: {selected_arm_name}")',
                'print(f"Validation-selected preprocessing: {selected_arm_name}")',
            )
        source = source.replace(
            "densenet121_preprocessing_quality_ablation",
            "densenet121_preprocessing_confirmation",
        )
        source = source.replace(
            "DenseNet-121 Preprocessing and Image-Quality Ablation",
            "DenseNet-121 Two-Arm Preprocessing Confirmation",
        )
        source = source.replace(
            "Final selected preprocessing",
            "Confirmation-selected preprocessing",
        )
        cell["source"] = source.splitlines(keepends=True)


def replace_markdown(notebook: dict) -> None:
    notebook["cells"][0]["source"] = [
        "# DenseNet-121 Two-Arm Preprocessing Confirmation\n",
        "\n",
        "This focused validation-only experiment repeats the two relevant preprocessing arms from scratch:\n",
        "\n",
        "- `current_pad_then_clahe2`: deployed order, square padding before LAB CLAHE `2.0`.\n",
        "- `clahe1_25_then_pad`: candidate order, LAB CLAHE `1.25` before square padding.\n",
        "\n",
        "Both arms use the same DenseNet-121 native-CAM architecture, CE loss, split, seed, sampler, augmentation, and 5/15/10 epoch schedule. The test split is never loaded. This run removes the interrupted-optimizer confound from the original six-arm ablation.\n",
    ]
    replacements = {
        "Five deterministic arms isolate enhancement strength and enhancement/padding order":
            "Two arms directly confirm enhancement strength and enhancement/padding order",
        "After those controls finish, the best validation-eligible deterministic arm\nreceives one follow-up with conservative random gamma, blur/sharpness, and\nnoise":
            "No robustness follow-up is trained; the earlier acquisition-robust arm was rejected",
        "Deterministic preprocessing arms:": "Confirmation preprocessing arms:",
        "## Shared CE Training and Frozen CAM Audit": "## Shared CE Training and Frozen CAM Audit",
        "Each arm resets seed 42": "Each of the two arms resets seed 42",
        "## Validation-Only Selection": "## Validation-Only Confirmation Decision",
        "Only arms within 0.01": "The winner is chosen by the same frozen rule: only arms within 0.01",
    }
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "markdown":
            continue
        source = "".join(cell.get("source", []))
        for old, new in replacements.items():
            source = source.replace(old, new)
        source = source.replace("robustness follow-up", "two-arm confirmation")
        cell["source"] = source.splitlines(keepends=True)
    notebook["cells"][1]["source"] = [
        "## Execution\n",
        "\n",
        "Run every cell from top to bottom in one A100 or T4 Colab runtime. The notebook trains exactly two complete 5/15/10-stage arms and saves checkpoints in one unique timestamped confirmation directory. Resume is automatic after interruption. Do not skip the preprocessing preview, CAM audit, comparison, report, or runtime-release cells.\n",
    ]
    notebook["cells"][13]["source"] = [
        "## Reproducible Report\n",
        "\n",
        "The report records both preprocessing methods, predictive metrics, CAM geometry, sharpness association, and the checkpoint eligible for a future locked-holdout evaluation.\n",
    ]
    notebook["cells"][15]["source"] = [
        "## References\n",
        "\n",
        "- Pizer SM et al. *Adaptive Histogram Equalization and Its Variations*. 1987. https://doi.org/10.1016/S0734-189X(87)80186-X\n",
        "- Tiulpin A et al. *Automatic Knee Osteoarthritis Diagnosis from Plain Radiographs*. 2018. https://doi.org/10.1038/s41598-018-20132-7\n",
        "- Moré LG et al. *Parameter Tuning of CLAHE Based on Multi-objective Optimization*. 2015. https://doi.org/10.1109/ICIP.2015.7351687\n",
        "\n",
        "These sources motivate controlled preprocessing; none establishes an optimal CLAHE setting for this exact dataset and ROI distribution. The winner is therefore selected empirically on validation only.\n",
    ]


def main() -> None:
    notebook = deepcopy(json.loads(SOURCE.read_text(encoding="utf-8")))
    replace_code(notebook)
    replace_markdown(notebook)
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
            ast.parse("".join(cell.get("source", [])))
    TARGET.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(TARGET)


if __name__ == "__main__":
    main()
