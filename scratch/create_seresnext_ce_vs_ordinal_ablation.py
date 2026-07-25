import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks/experiments/se_resnext50_32x4d_cam_ablation.ipynb"
TARGET = ROOT / "notebooks/experiments/se_resnext50_32x4d_ce_vs_ordinal_loss.ipynb"


def source(cell):
    return "".join(cell["source"])


def set_source(cell, value):
    cell["source"] = value.splitlines(keepends=True)


def replace_once(value, old, new):
    if value.count(old) != 1:
        raise RuntimeError(f"Expected one occurrence, found {value.count(old)}: {old[:80]!r}")
    return value.replace(old, new, 1)


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
cells = notebook["cells"]

set_source(
    cells[0],
    """# SE-ResNeXt-50: CE versus Ordinal Soft-Label Loss

This validation-only ablation isolates the loss target while holding the backbone,
five-map native-CAM head, natural-orientation preprocessing, augmentation, sampler,
initialization seed, optimizer schedule, and checkpoint-selection score fixed.

The two arms are:

1. standard hard-label cross-entropy;
2. Gaussian ordinal soft-label cross-entropy with `sigma=0.70`.

The ordinal arm preserves five mutually exclusive grade logits and one native class
map per KL grade. CORN is intentionally excluded because its four conditional
threshold outputs would change the head and require a different grade-CAM definition.
The test split is never loaded. Run every cell from top to bottom in a fresh Colab
GPU runtime.
""",
)

config = source(cells[4])
config = replace_once(
    config,
    'architecture = "seresnext_cam_comparison_harness"',
    'architecture = "seresnext_ce_vs_ordinal_soft_label"',
)
config = replace_once(
    config,
    'checkpoint_root, f"{run_timestamp}_seresnext_cam_comparison"',
    'checkpoint_root, f"{run_timestamp}_seresnext_ce_vs_ordinal_soft_label"',
)
config = replace_once(
    config,
    "canonicalize_laterality = True    # Mirror R knees so all joints share one orientation",
    "canonicalize_laterality = False   # Preserve natural orientation at train and inference",
)
config = replace_once(
    config,
    "    contrast_jitter = 0.08\n",
    "    contrast_jitter = 0.08\n"
    "    horizontal_flip_probability = 0.50\n"
    "    gamma_probability = 0.20\n"
    "    gamma_range = (0.90, 1.10)\n",
)
config = replace_once(
    config,
    "    # Native grade-specific maps require one CE logit/map per KL class.\n",
    "    # Both losses retain one logit and one native class map per KL grade.\n",
)
config = config.replace("os.makedirs(CHECKPOINT_SAVE_DIR, exist_ok=False)\n", "")
set_source(cells[4], config)

compatibility = source(cells[6]).replace(
    "os.makedirs(CHECKPOINT_SAVE_DIR, exist_ok=True)\n", ""
)
set_source(cells[6], compatibility)

transforms_code = source(cells[8])
gamma_class = '''class RandomGammaCorrection:
    """Apply mild gamma correction with torchvision-compatible primitives."""

    def __init__(self, gamma_range=(0.90, 1.10), probability=0.20):
        low, high = gamma_range
        if low <= 0 or high < low:
            raise ValueError("gamma_range must contain positive ordered values")
        self.gamma_range = (float(low), float(high))
        self.probability = float(probability)

    def __call__(self, image):
        if torch.rand(1).item() >= self.probability:
            return image
        gamma = torch.empty(1).uniform_(*self.gamma_range).item()
        return transforms.functional.adjust_gamma(image, gamma=gamma, gain=1.0)


'''
transforms_code = replace_once(
    transforms_code,
    "def get_transforms(img_size=400, crop_size=384):\n",
    gamma_class + "def get_transforms(img_size=400, crop_size=384):\n",
)
transforms_code = replace_once(
    transforms_code,
    '    """Build transforms after laterality canonicalization; no random mirroring."""',
    '    """Build natural-orientation transforms shared by both loss arms."""',
)
transforms_code = transforms_code.replace(
    "        transforms.ToPILImage(),\n"
    "        transforms.RandomRotation(degrees=TrainingConfig.rotation_degrees),",
    "        transforms.ToPILImage(),\n"
    "        transforms.RandomHorizontalFlip(\n"
    "            p=TrainingConfig.horizontal_flip_probability\n"
    "        ),\n"
    "        transforms.RandomRotation(degrees=TrainingConfig.rotation_degrees),",
)
transforms_code = transforms_code.replace(
    "        transforms.ColorJitter(\n"
    "            brightness=TrainingConfig.brightness_jitter,\n"
    "            contrast=TrainingConfig.contrast_jitter,\n"
    "        ),\n"
    "        transforms.Resize((img_size, img_size)),",
    "        transforms.ColorJitter(\n"
    "            brightness=TrainingConfig.brightness_jitter,\n"
    "            contrast=TrainingConfig.contrast_jitter,\n"
    "        ),\n"
    "        RandomGammaCorrection(\n"
    "            gamma_range=TrainingConfig.gamma_range,\n"
    "            probability=TrainingConfig.gamma_probability,\n"
    "        ),\n"
    "        transforms.Resize((img_size, img_size)),",
)
set_source(cells[8], transforms_code)

dataset_code = source(cells[10])
dataset_code = dataset_code.replace(
    '    """Load one split and apply identical laterality logic to every downstream path."""',
    '    """Load one split while preserving each image\'s natural orientation."""',
)
set_source(cells[10], dataset_code)

set_source(
    cells[12],
    '''# Build datasets only. Each arm receives a fresh sampler and loader with the same seed.
from torch.utils.data import WeightedRandomSampler


train_transform, val_transform, _, minority_train_transform = get_transforms(
    img_size=TrainingConfig.img_size,
    crop_size=TrainingConfig.crop_size,
)
minority_transform = (
    minority_train_transform if TrainingConfig.use_minority_aug else None
)

train_dataset = KaggleKneeOsteoarthritisDataset(
    root=DATASET_ROOT_PATH,
    split_dir="train",
    transform=train_transform,
    minority_transform=minority_transform,
)
train_hashes = set(train_dataset.image_hashes)

val_split_dir = "val"
if not os.path.isdir(os.path.join(DATASET_ROOT_PATH, val_split_dir)):
    raise FileNotFoundError("A validation split is required; test fallback is disabled.")
val_dataset = KaggleKneeOsteoarthritisDataset(
    root=DATASET_ROOT_PATH,
    split_dir=val_split_dir,
    transform=val_transform,
    exclude_hashes=train_hashes,
)

class_counts = Counter(train_dataset.labels)
print(f"Training class distribution: {dict(sorted(class_counts.items()))}")
print(f"Validation images: {len(val_dataset)}")
''',
)

experiment = source(cells[14])
old_arms = '''    "arms": [
        {"name": "multiscale_mlp_hirescam_ce", "architecture": "multiscale_mlp", "loss": "ce"},
        {"name": "final_native_cam_ce", "architecture": "final_linear_cam", "loss": "ce"},
        {"name": "final_native_cam_joint_guided_005", "architecture": "final_linear_cam", "loss": "ce", "joint_guidance_weight": 0.05},
        {"name": "final_native_cam_softlabel", "architecture": "final_linear_cam", "loss": "ordinal_soft_label"},
        {"name": "fpn_native_cam_ce", "architecture": "fpn_cam", "loss": "ce"},
        {"name": "fpn_native_cam_softlabel", "architecture": "fpn_cam", "loss": "ordinal_soft_label"},
    ],'''
new_arms = '''    "arms": [
        {
            "name": "final_native_cam_ce",
            "architecture": "final_linear_cam",
            "loss": "ce",
        },
        {
            "name": "final_native_cam_ordinal_soft_label",
            "architecture": "final_linear_cam",
            "loss": "ordinal_soft_label",
        },
    ],'''
experiment = replace_once(experiment, old_arms, new_arms)
experiment = replace_once(
    experiment,
    'Path(TrainingConfig.checkpoint_root) / "seresnext50_cam_ablations" / stamp',
    'Path(TrainingConfig.checkpoint_root) / "seresnext50_loss_ablations" / stamp',
)
set_source(cells[14], experiment)

report_code = source(cells[18])
report_code = report_code.replace(
    "# SE-ResNeXt-50 Metric and Faithful-CAM Ablation",
    "# SE-ResNeXt-50 CE versus Ordinal Soft-Label Ablation",
)
report_code = report_code.replace(
    "Laterality was canonicalized before augmentation. All arms used the same validation cases, full inverse-frequency sampler, 5/15/10 schedule, and predictive score.",
    "Natural orientation was preserved. Both arms used the same training-only horizontal flip and mild gamma correction, validation cases, full inverse-frequency sampler, initialization seed, 5/15/10 schedule, and predictive score.",
)
report_code = report_code.replace(
    'SERESNEXT_BATCH_DIR / "seresnext_cam_ablation_report.md"',
    'SERESNEXT_BATCH_DIR / "seresnext_ce_vs_ordinal_report.md"',
)
report_code = report_code.replace(
    "SERESNEXT_BATCH_DIR / 'seresnext_cam_ablation_report.md'",
    "SERESNEXT_BATCH_DIR / 'seresnext_ce_vs_ordinal_report.md'",
)
set_source(cells[18], report_code)

gradcam_code = source(cells[20])
gradcam_code = gradcam_code.replace(
    "The native map is the production explanation because its spatial mean is the exact predicted CE logit.",
    "The native map is the production explanation because its spatial mean is the exact predicted grade logit for either target construction.",
)
gradcam_code = gradcam_code.replace(
    'SERESNEXT_BATCH_DIR / "seresnext_cam_ablation_report.md"',
    'SERESNEXT_BATCH_DIR / "seresnext_ce_vs_ordinal_report.md"',
)
set_source(cells[20], gradcam_code)

set_source(
    cells[22],
    source(cells[22]).replace(
        "SE-ResNeXt CAM ablation complete.",
        "SE-ResNeXt CE-versus-ordinal ablation complete.",
    ),
)

set_source(cells[13], "## Model and Loss Comparison\n")
set_source(cells[15], "## Same-Case Native-CAM Localization and Faithfulness Audit\n")
set_source(cells[17], "## Selection Report and Paired Validation Bootstrap\n")
set_source(cells[19], "## Native CAM versus Final-Layer Grad-CAM Sanity Check\n")

for cell in cells:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

notebook.setdefault("metadata", {}).setdefault("kernelspec", {})["display_name"] = "Python 3"
TARGET.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(TARGET)
