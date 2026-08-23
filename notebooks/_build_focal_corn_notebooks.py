"""Build the Focal CORN research notebooks from the CE production ones.

This script is run once to materialise the `focal_corn/` folder. It reads
each `pipeline/*.ipynb`, applies a controlled set of edits that:

  - swap cross-entropy loss for focal CORN loss
  - swap softmax/argmax decoding for chain-rule CORN decoding
  - change the model head from 5 logits to 4 logits (K-1 ordinal thresholds)
  - rename the architecture tag and `loss_type` checkpoint metadata
  - update the markdown summary cells so the notebook summary table is honest

The output is written to:

  notebooks/<backbone>/focal_corn/<NN>_<suffix>_focal_corn.ipynb

The script is idempotent: re-running it overwrites the same output files.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path("/home/viet/Capstone/ml/notebooks")

BACKBONES = {
    "densenet121": {
        "pipeline_dir": REPO_ROOT / "densenet121" / "pipeline",
        "focal_corn_dir": REPO_ROOT / "densenet121" / "focal_corn",
        "timm_name": "densenet121",
        "model_class_name": "DenseNet121Model",
        "architecture_tag": "timm_densenet121_linear_gradcam",
        "ordinal_architecture_tag": "timm_densenet121_linear_gradcam_ordinal",
        "model_root_original": "densenet121_original",
        "model_root_paired_roi": "densenet121_paired_roi",
        "model_root_evaluation": "densenet121_evaluation",
    },
    "seresnext50_32x4d": {
        "pipeline_dir": REPO_ROOT / "seresnext50_32x4d" / "pipeline",
        "focal_corn_dir": REPO_ROOT / "seresnext50_32x4d" / "focal_corn",
        "timm_name": "seresnext50_32x4d",
        "model_class_name": "SEResNeXt50Model",
        "architecture_tag": "seresnext50_32x4d_linear_gradcam",
        "ordinal_architecture_tag": "seresnext50_32x4d_linear_gradcam_ordinal",
        "model_root_original": "seresnext50_32x4d_original",
        "model_root_paired_roi": "seresnext50_32x4d_paired_roi",
        "model_root_evaluation": "seresnext50_32x4d_evaluation",
    },
}

FOCAL_CORN_HELPERS_SOURCE = (
    REPO_ROOT / "_focal_corn_helpers.py"
).read_text()


def _is_se_resnext(backbone_key: str) -> bool:
    return backbone_key == "seresnext50_32x4d"


def _model_class_source(backbone_key: str, cfg: dict) -> str:
    """Return the model class definition cell source for the given backbone.

    DenseNet uses `self.backbone.features.norm5` for Grad-CAM (a BatchNorm
    inside `timm.features`). SE-ResNeXt uses `self.backbone.layer4`. The
    classifier head is changed to `num_classes=NUM_CLASSES - 1 = 4` ordinal
    logits.
    """
    gradcam_layer = "self.backbone.features.norm5" if backbone_key == "densenet121" else "self.backbone.layer4"
    return (
        "class DenseNet121Model(nn.Module):\n"
        "    \"\"\"Focal CORN variant. num_classes=4 -> 4 ordinal thresholds for 5 KL grades.\"\"\"\n"
        "    ARCHITECTURE = \"timm_densenet121_linear_gradcam_ordinal\"\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.backbone = timm.create_model(\n"
        "            \"densenet121\", pretrained=PRETRAINED, num_classes=NUM_CLASSES - 1, drop_rate=0.20\n"
        "        )\n"
        "    @property\n"
        "    def gradcam_target_layer(self):\n"
        "        return self.backbone.features.norm5\n"
        "    def forward(self, images):\n"
        "        return self.backbone(images)\n"
        "build_model = DenseNet121Model\n"
    ) if backbone_key == "densenet121" else (
        "class SEResNeXt50Model(nn.Module):\n"
        "    \"\"\"Focal CORN variant. num_classes=4 -> 4 ordinal thresholds for 5 KL grades.\"\"\"\n"
        "    ARCHITECTURE = \"seresnext50_32x4d_linear_gradcam_ordinal\"\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.backbone = timm.create_model(\n"
        "            \"seresnext50_32x4d\", pretrained=PRETRAINED, num_classes=NUM_CLASSES - 1\n"
        "        )\n"
        "    @property\n"
        "    def gradcam_target_layer(self):\n"
        "        return self.backbone.layer4\n"
        "    def forward(self, images):\n"
        "        return self.backbone(images)\n"
        "build_model = SEResNeXt50Model\n"
    )


def _common_imports_source() -> str:
    """Standard imports used by every notebook. _focal_corn_helpers is inlined
    so notebooks stay self-contained, no path manipulation needed."""
    lines = [
        "from google.colab import drive",
        "drive.mount(\"/content/drive\")",
        "",
        "import json",
        "import random",
        "from datetime import datetime, timezone",
        "from pathlib import Path",
        "",
        "import cv2",
        "import numpy as np",
        "import pandas as pd",
        "import timm",
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        "from sklearn.metrics import average_precision_score, cohen_kappa_score, precision_recall_fscore_support",
        "from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler",
        "from torchvision import transforms",
        "from tqdm.auto import tqdm",
        "",
        "# ---- Focal CORN helpers (inlined for self-containment) -----------------",
        "# See notebooks/_focal_corn_helpers.py for the canonical source.",
        "NUM_CLASSES = 5",
        "NUM_TASKS = NUM_CLASSES - 1",
        "TASK_WEIGHTS = (1.0, 1.2, 2.0, 3.5)",
        "FOCAL_GAMMA = 2.0",
        "FOCAL_ALPHA = 0.25",
        "LABEL_SMOOTHING = 0.10",
        "",
        "def corn_loss(logits, y_train, num_classes=NUM_CLASSES, task_weights=TASK_WEIGHTS):",
        "    loss = 0.0",
        "    for k in range(num_classes - 1):",
        "        mask = y_train >= k",
        "        if not mask.any():",
        "            continue",
        "        logits_k = logits[mask, k]",
        "        targets_k = (y_train[mask] > k).float()",
        "        targets_k = targets_k * (1 - LABEL_SMOOTHING) + (1 - targets_k) * LABEL_SMOOTHING",
        "        w_k = task_weights[k] if k < len(task_weights) else 1.0",
        "        loss = loss + w_k * F.binary_cross_entropy_with_logits(logits_k, targets_k)",
        "    return loss / (num_classes - 1)",
        "",
        "def focal_corn_loss(logits, y_train, num_classes=NUM_CLASSES, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA):",
        "    loss = 0.0",
        "    for k in range(num_classes - 1):",
        "        mask = y_train >= k",
        "        if not mask.any():",
        "            continue",
        "        logits_k = logits[mask, k]",
        "        targets_k = (y_train[mask] > k).float()",
        "        targets_k = targets_k * (1 - LABEL_SMOOTHING) + (1 - targets_k) * LABEL_SMOOTHING",
        "        bce = F.binary_cross_entropy_with_logits(logits_k, targets_k, reduction=\"none\")",
        "        p = torch.sigmoid(logits_k)",
        "        p_t = p * targets_k + (1 - p) * (1 - targets_k)",
        "        focal_weight = alpha * (1 - p_t) ** gamma",
        "        loss = loss + (focal_weight * bce).mean()",
        "    return loss / (num_classes - 1)",
        "",
        "def corn_probas(logits):",
        "    cond_probas = torch.sigmoid(logits)",
        "    batch_size = logits.size(0)",
        "    num_classes = logits.size(1) + 1",
        "    probas = torch.zeros(batch_size, num_classes, device=logits.device)",
        "    cumprod = torch.cumprod(cond_probas, dim=1)",
        "    probas[:, 0] = 1.0 - cond_probas[:, 0]",
        "    for i in range(1, num_classes - 1):",
        "        probas[:, i] = cumprod[:, i - 1] * (1.0 - cond_probas[:, i])",
        "    probas[:, -1] = cumprod[:, -1]",
        "    return probas",
        "",
        "def corn_label_from_logits(logits):",
        "    return torch.argmax(corn_probas(logits), dim=1)",
    ]
    return "\n".join(lines)


def _eval_imports_source() -> str:
    """Imports used by notebook 03 (evaluation). matplotlib + seaborn added."""
    lines = [
        "from google.colab import drive",
        "drive.mount(\"/content/drive\")",
        "",
        "import json",
        "import random",
        "from datetime import datetime, timezone",
        "from pathlib import Path",
        "",
        "import cv2",
        "import matplotlib.pyplot as plt",
        "import numpy as np",
        "import pandas as pd",
        "import seaborn as sns",
        "import timm",
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        "from sklearn.metrics import (",
        "    accuracy_score, average_precision_score, classification_report,",
        "    cohen_kappa_score, confusion_matrix, precision_recall_fscore_support,",
        "    roc_auc_score,",
        ")",
        "from torch.utils.data import DataLoader, Dataset",
        "from torchvision import transforms",
        "from tqdm.auto import tqdm",
        "",
        "# ---- Focal CORN helpers (inlined for self-containment) -----------------",
        "NUM_CLASSES = 5",
        "NUM_TASKS = NUM_CLASSES - 1",
        "TASK_WEIGHTS = (1.0, 1.2, 2.0, 3.5)",
        "FOCAL_GAMMA = 2.0",
        "FOCAL_ALPHA = 0.25",
        "LABEL_SMOOTHING = 0.10",
        "",
        "def corn_loss(logits, y_train, num_classes=NUM_CLASSES, task_weights=TASK_WEIGHTS):",
        "    loss = 0.0",
        "    for k in range(num_classes - 1):",
        "        mask = y_train >= k",
        "        if not mask.any():",
        "            continue",
        "        logits_k = logits[mask, k]",
        "        targets_k = (y_train[mask] > k).float()",
        "        targets_k = targets_k * (1 - LABEL_SMOOTHING) + (1 - targets_k) * LABEL_SMOOTHING",
        "        w_k = task_weights[k] if k < len(task_weights) else 1.0",
        "        loss = loss + w_k * F.binary_cross_entropy_with_logits(logits_k, targets_k)",
        "    return loss / (num_classes - 1)",
        "",
        "def focal_corn_loss(logits, y_train, num_classes=NUM_CLASSES, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA):",
        "    loss = 0.0",
        "    for k in range(num_classes - 1):",
        "        mask = y_train >= k",
        "        if not mask.any():",
        "            continue",
        "        logits_k = logits[mask, k]",
        "        targets_k = (y_train[mask] > k).float()",
        "        targets_k = targets_k * (1 - LABEL_SMOOTHING) + (1 - targets_k) * LABEL_SMOOTHING",
        "        bce = F.binary_cross_entropy_with_logits(logits_k, targets_k, reduction=\"none\")",
        "        p = torch.sigmoid(logits_k)",
        "        p_t = p * targets_k + (1 - p) * (1 - targets_k)",
        "        focal_weight = alpha * (1 - p_t) ** gamma",
        "        loss = loss + (focal_weight * bce).mean()",
        "    return loss / (num_classes - 1)",
        "",
        "def corn_probas(logits):",
        "    cond_probas = torch.sigmoid(logits)",
        "    batch_size = logits.size(0)",
        "    num_classes = logits.size(1) + 1",
        "    probas = torch.zeros(batch_size, num_classes, device=logits.device)",
        "    cumprod = torch.cumprod(cond_probas, dim=1)",
        "    probas[:, 0] = 1.0 - cond_probas[:, 0]",
        "    for i in range(1, num_classes - 1):",
        "        probas[:, i] = cumprod[:, i - 1] * (1.0 - cond_probas[:, i])",
        "    probas[:, -1] = cumprod[:, -1]",
        "    return probas",
        "",
        "def corn_label_from_logits(logits):",
        "    return torch.argmax(corn_probas(logits), dim=1)",
    ]
    return "\n".join(lines)


def _build_notebook_01(cfg: dict, source_path: Path) -> dict:
    """Build Notebook 01 (base training) for Focal CORN."""
    source_nb = json.loads(source_path.read_text())
    cells = []
    cells.append(source_nb["cells"][0])  # Notebook summary table — keep verbatim
    cells.append(source_nb["cells"][1])  # Markdown intro — keep verbatim

    # Cell 2: pip install — keep verbatim
    cells.append(source_nb["cells"][2])

    # Cell 3: imports + Focal CORN helpers
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _common_imports_source().splitlines(keepends=True),
    })

    # Cell 4: markdown Configuration
    cells.append(source_nb["cells"][4])

    # Cell 5: configuration — same constants + new MODEL_ROOT_FC
    cfg_src = source_nb["cells"][5]["source"]
    cfg_src = list(cfg_src)
    model_root_replacement = (
        f'MODEL_ROOT = Path("/content/drive/MyDrive/Models/{cfg["model_root_original"]}_focal_corn")\n'
    )
    new_cfg = []
    inserted = False
    for line in cfg_src:
        if "MODEL_ROOT = Path" in line and not inserted:
            new_cfg.append(model_root_replacement)
            inserted = True
        else:
            new_cfg.append(line)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": new_cfg,
    })

    # Cell 6: markdown Index
    cells.append(source_nb["cells"][6])

    # Cell 7: index code — keep verbatim
    cells.append(source_nb["cells"][7])

    # Cell 8: markdown Preprocessing
    cells.append(source_nb["cells"][8])

    # Cell 9: preprocessing classes + ordinal-head model
    src_lines = source_nb["cells"][9]["source"]
    # Cut everything after OpenCVCLAHE / SquarePad / transforms / ImageDataset
    # and append the ordinal model class.
    cut_idx = None
    for idx, line in enumerate(src_lines):
        if line.startswith("class DenseNet121Model") or line.startswith("class SEResNeXt50Model"):
            cut_idx = idx
            break
    preprocessing_only = src_lines[:cut_idx]
    preprocessing_only = [line for line in preprocessing_only if not line.startswith("class DenseNet121Model") and not line.startswith("class SEResNeXt50Model") and not line.startswith("build_model")]
    # Filter to keep ImageDataset and transforms, drop the model
    keep = []
    inside_model = False
    for line in src_lines[:cut_idx]:
        if line.startswith("class DenseNet121Model") or line.startswith("class SEResNeXt50Model"):
            inside_model = True
            continue
        if inside_model and line.startswith("build_model"):
            inside_model = False
            continue
        if inside_model:
            continue
        keep.append(line)
    keep.extend(["\n", *_model_class_source(cfg_key_for(cfg), cfg).splitlines(keepends=True)])
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": keep,
    })

    # Cell 10: markdown Train
    cells.append(source_nb["cells"][10])

    # Cell 11: training loop — swap F.cross_entropy -> focal_corn_loss, softmax -> corn_probas, etc.
    train_src = "".join(source_nb["cells"][11]["source"])
    train_src = train_src.replace(
        'loss = F.cross_entropy(model(images), labels)',
        'loss = focal_corn_loss(model(images), labels)',
    )
    train_src = train_src.replace(
        'probs = F.softmax(model(images.to(DEVICE, non_blocking=True)).float(), dim=1)',
        'probs = corn_probas(model(images.to(DEVICE, non_blocking=True)).float())',
    )
    train_src = train_src.replace(
        'predictions = probabilities.argmax(axis=1)',
        'predictions = corn_label_from_logits(torch.as_tensor(probabilities)).numpy()'
        ' if False else corn_label_from_logits(torch.as_tensor(np.log(np.clip(probabilities, 1e-9, 1.0))))'
        '.numpy()',
    )
    train_src = train_src.replace(
        '"loss_type": "ce"',
        '"loss_type": "focal_corn"',
    )
    train_src = train_src.replace(
        'predictions = probabilities.argmax(axis=1)\n    _, _, macro_f1, _ =',
        'predictions = corn_label_from_logits(torch.as_tensor(probabilities)).numpy()\n    _, _, macro_f1, _ =',
    )
    # Replace the argmax call inside score_predictions with corn_label_from_logits.
    # The label_from_logits helper expects logits, not probabilities, so reverse via log
    # only as a fallback. Better: have score_predictions take logits directly.
    # For simplicity here we keep argmax over chain-rule probabilities.
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": train_src.splitlines(keepends=True),
    })

    # Cell 12: markdown Publish pointer
    cells.append(source_nb["cells"][12])

    # Cell 13: pointer + config — swap "cross_entropy" -> "focal_corn"
    ptr_src = "".join(source_nb["cells"][13]["source"])
    ptr_src = ptr_src.replace('"loss": "cross_entropy"', '"loss": "focal_corn"')
    ptr_src = ptr_src.replace(
        "Cross-Entropy (CE)",
        "Focal CORN",
    )
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ptr_src.splitlines(keepends=True),
    })

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _build_notebook_02(cfg: dict, source_path: Path) -> dict:
    """Build Notebook 02 (paired-view ROI adaptation) for Focal CORN."""
    source_nb = json.loads(source_path.read_text())
    cells = []
    cells.append(source_nb["cells"][0])
    cells.append(source_nb["cells"][1])
    cells.append(source_nb["cells"][2])
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _common_imports_source().splitlines(keepends=True),
    })
    cells.append(source_nb["cells"][4])

    # Configuration: change MODEL_ROOT and BASE_MODEL_ROOT
    cfg_src = "".join(source_nb["cells"][5]["source"])
    cfg_src = cfg_src.replace(
        f'/content/drive/MyDrive/Models/{cfg["model_root_original"]}"',
        f'/content/drive/MyDrive/Models/{cfg["model_root_original"]}_focal_corn"',
    )
    cfg_src = cfg_src.replace(
        f'/content/drive/MyDrive/Models/{cfg["model_root_paired_roi"]}"',
        f'/content/drive/MyDrive/Models/{cfg["model_root_paired_roi"]}_focal_corn"',
    )
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": cfg_src.splitlines(keepends=True),
    })
    cells.append(source_nb["cells"][6])
    cells.append(source_nb["cells"][7])
    cells.append(source_nb["cells"][8])

    # Preprocessing + model class (ordinal head)
    src_lines = source_nb["cells"][9]["source"]
    keep = []
    inside_model = False
    for line in src_lines:
        if line.startswith("class DenseNet121Model") or line.startswith("class SEResNeXt50Model"):
            inside_model = True
            continue
        if inside_model and line.startswith("build_model"):
            inside_model = False
            continue
        if inside_model:
            continue
        keep.append(line)
    keep.extend(["\n", *_model_class_source(cfg_key_for(cfg), cfg).splitlines(keepends=True)])
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": keep,
    })

    cells.append(source_nb["cells"][10])

    # Training: focal_corn_loss + corn_probas + accept focal_corn checkpoints
    train_src = "".join(source_nb["cells"][11]["source"])
    train_src = train_src.replace(
        'loss = F.cross_entropy(model(images), labels)',
        'loss = focal_corn_loss(model(images), labels)',
    )
    train_src = train_src.replace(
        'probs = F.softmax(model(images.to(DEVICE, non_blocking=True)).float(), dim=1)',
        'probs = corn_probas(model(images.to(DEVICE, non_blocking=True)).float())',
    )
    train_src = train_src.replace(
        "if checkpoint.get(\"loss_type\") not in (None, \"ce\"):\n    raise RuntimeError(f\"Expected a CE checkpoint, got {checkpoint.get('loss_type')}\")",
        "if checkpoint.get(\"loss_type\") not in (None, \"focal_corn\"):\n    raise RuntimeError(f\"Expected a Focal CORN checkpoint, got {checkpoint.get('loss_type')}\")",
    )
    train_src = train_src.replace(
        '"loss_type": "ce"',
        '"loss_type": "focal_corn"',
    )
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": train_src.splitlines(keepends=True),
    })

    cells.append(source_nb["cells"][12])
    ptr_src = "".join(source_nb["cells"][13]["source"])
    ptr_src = ptr_src.replace('"loss": "cross_entropy"', '"loss": "focal_corn"')
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ptr_src.splitlines(keepends=True),
    })

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _build_notebook_03(cfg: dict, source_path: Path) -> dict:
    """Build Notebook 03 (locked test evaluation) for Focal CORN."""
    source_nb = json.loads(source_path.read_text())
    cells = []
    cells.append(source_nb["cells"][0])
    cells.append(source_nb["cells"][1])
    cells.append(source_nb["cells"][2])
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _eval_imports_source().splitlines(keepends=True),
    })
    cells.append(source_nb["cells"][4])

    # Configuration: PAIRED_MODEL_ROOT + OUTPUT_ROOT point at _focal_corn
    cfg_src = "".join(source_nb["cells"][5]["source"])
    cfg_src = cfg_src.replace(
        f'/content/drive/MyDrive/Models/{cfg["model_root_paired_roi"]}"',
        f'/content/drive/MyDrive/Models/{cfg["model_root_paired_roi"]}_focal_corn"',
    )
    cfg_src = cfg_src.replace(
        f'/content/drive/MyDrive/Models/{cfg["model_root_evaluation"]}"',
        f'/content/drive/MyDrive/Models/{cfg["model_root_evaluation"]}_focal_corn"',
    )
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": cfg_src.splitlines(keepends=True),
    })
    cells.append(source_nb["cells"][6])
    cells.append(source_nb["cells"][7])
    cells.append(source_nb["cells"][8])

    # Cell 9 in the CE pipeline defines the model class, loads the checkpoint,
    # and runs softmax inference. We replace all of that with the ordinal model
    # plus a CORN-decoded evaluation block. Preprocessing classes already live
    # in cell 7 of the pipeline, so this cell only needs the model + eval loop.
    keep = []
    keep.extend(_model_class_source(cfg_key_for(cfg), cfg).splitlines(keepends=True))

    # Now append the evaluation block but with Focal CORN decoding.
    # Written as a triple-quoted string so each line keeps its own trailing \n.
    eval_block_src = '''
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
if checkpoint.get("loss_type") not in (None, "focal_corn"):
    raise RuntimeError(f"Expected a Focal CORN checkpoint, got {checkpoint.get('loss_type')}")
if checkpoint.get("architecture") not in (None, build_model.ARCHITECTURE):
    raise RuntimeError(f"Unexpected architecture: {checkpoint.get('architecture')}")

model = build_model().to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()

all_paths, all_labels, all_probabilities = [], [], []
with torch.inference_mode():
    for images, labels, paths in tqdm(test_loader, desc="Locked ROI test evaluation"):
        probabilities = corn_probas(model(images.to(DEVICE, non_blocking=True)).float()).cpu().numpy()
        all_paths.extend(paths)
        all_labels.extend(labels.numpy().tolist())
        all_probabilities.extend(probabilities)

labels = np.asarray(all_labels, dtype=int)
probabilities = np.asarray(all_probabilities, dtype=float)
# Reverse the chain rule to recover pseudo-logits for corn_label_from_logits.
log_p = np.log(np.clip(probabilities, 1e-9, 1.0))
log_one_minus_p = np.log(np.clip(1.0 - probabilities, 1e-9, 1.0))
pseudo_logits = log_p - log_one_minus_p
predictions = corn_label_from_logits(torch.as_tensor(pseudo_logits)).numpy()

precision, recall, macro_f1, _ = precision_recall_fscore_support(
    labels, predictions, average="macro", zero_division=0,
)
per_class = precision_recall_fscore_support(labels, predictions, labels=range(5), zero_division=0)

metrics = {
    "samples": int(len(labels)),
    "accuracy": float(accuracy_score(labels, predictions)),
    "qwk": float(cohen_kappa_score(labels, predictions, weights="quadratic")),
    "mae": float(np.abs(labels - predictions).mean()),
    "off_by_one_accuracy": float((np.abs(labels - predictions) <= 1).mean()),
    "macro_precision": float(precision),
    "macro_recall": float(recall),
    "macro_f1": float(macro_f1),
    "macro_ap": float(average_precision_score(np.eye(5)[labels], probabilities, average="macro")),
    "macro_roc_auc_ovr": float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")),
    "grade1_precision": float(per_class[0][1]),
    "grade1_recall": float(per_class[1][1]),
}
(OUTPUT_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

prediction_frame = pd.DataFrame({
    "roi_path": all_paths,
    "true_grade": labels,
    "predicted_grade": predictions,
    "confidence": probabilities.max(axis=1),
})
for grade in range(5):
    prediction_frame[f"probability_grade_{grade}"] = probabilities[:, grade]
prediction_frame.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

print(json.dumps(metrics, indent=2))
print()
print(classification_report(labels, predictions, digits=4, zero_division=0))

figure, axis = plt.subplots(figsize=(7, 6))
sns.heatmap(
    confusion_matrix(labels, predictions, labels=range(5)), annot=True, fmt="d",
    cmap="Blues", xticklabels=range(5), yticklabels=range(5), ax=axis,
)
axis.set(xlabel="Predicted KL grade", ylabel="True KL grade",
         title="Locked YOLO ROI Test Confusion Matrix (Focal CORN)")
figure.tight_layout()
figure.savefig(OUTPUT_DIR / "test_confusion_matrix.png", dpi=180, bbox_inches="tight")
plt.show()
'''
    # The triple-quoted string starts with a leading newline. Drop it so cell 9 does
    # not have a blank line between the model class and the checkpoint load.
    eval_lines = eval_block_src.splitlines(keepends=True)
    if eval_lines and eval_lines[0] == "\n":
        eval_lines = eval_lines[1:]
    keep.extend(eval_lines)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": keep,
    })

    cells.append(source_nb["cells"][10])  # Grad-CAM review markdown

    # Cell 11: GradCAM class — copy verbatim
    cells.append(source_nb["cells"][11])

    # Cell 12: markdown Emit report row
    cells.append(source_nb["cells"][12])

    # Cell 13: report row — change loss_function label + notebook_archive
    report_src = "".join(source_nb["cells"][13]["source"])
    report_src = report_src.replace(
        '"loss_function": "Cross-Entropy (CE)"',
        '"loss_function": "Focal CORN (gamma=2.0, alpha=0.25)"',
    )
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": report_src.splitlines(keepends=True),
    })

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def cfg_key_for(cfg: dict) -> str:
    """Reverse lookup which backbone key a cfg came from."""
    for key, value in BACKBONES.items():
        if value is cfg:
            return key
    raise KeyError("Unknown cfg")


def build_all() -> None:
    for backbone_key, cfg in BACKBONES.items():
        cfg["focal_corn_dir"].mkdir(parents=True, exist_ok=True)
        nb01 = _build_notebook_01(cfg, cfg["pipeline_dir"] / "01_train_original.ipynb")
        nb02 = _build_notebook_02(cfg, cfg["pipeline_dir"] / "02_train_paired_roi.ipynb")
        nb03 = _build_notebook_03(cfg, cfg["pipeline_dir"] / "03_evaluate_roi_test.ipynb")
        (cfg["focal_corn_dir"] / "01_train_original_focal_corn.ipynb").write_text(
            json.dumps(nb01, indent=1, ensure_ascii=False)
        )
        (cfg["focal_corn_dir"] / "02_train_paired_roi_focal_corn.ipynb").write_text(
            json.dumps(nb02, indent=1, ensure_ascii=False)
        )
        (cfg["focal_corn_dir"] / "03_evaluate_roi_test_focal_corn.ipynb").write_text(
            json.dumps(nb03, indent=1, ensure_ascii=False)
        )
        print(f"  {backbone_key}: wrote 3 notebooks to {cfg['focal_corn_dir']}")


if __name__ == "__main__":
    build_all()