import json

# Define the new cell contents
cell_1_code = """import os
import hashlib
from collections import Counter
from typing import List, Union
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import tqdm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Try mounting drive (if on Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted successfully.")
except ImportError:
    print("Not running in Google Colab. Skipping Drive mount.")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Install timm if needed
try:
    import timm
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "-q", "timm"])
    import timm

# Install torchmetrics if needed
try:
    import torchmetrics
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "-q", "torchmetrics"])
    import torchmetrics

# Install seaborn if needed
try:
    import seaborn
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "-q", "seaborn"])
    import seaborn"""

cell_2_code = """import subprocess
import os

# Unzip dataset from Drive if running on Google Colab
dataset_zip = "/content/drive/MyDrive/Datasets/kaggle_knee_osteoarthritis.zip"
if os.path.exists(dataset_zip):
    print("Unzipping dataset from Google Drive...")
    subprocess.run(["unzip", "-q", dataset_zip, "-d", "/content/Datasets"])
else:
    print("Zip file not found at default Drive path. Assuming local dataset path.")

# =========================================================================
# CONFIGURATION & PARAMETERS (Unified Input Config)
# =========================================================================
class TrainingConfig:
    # Model Selection
    model_name = "densenet201"          # options: "densenet201" (densenet201 is the default for this notebook)
    pretrained = True
    dropout_rate = 0.5
    
    # Dataset & Paths
    dataset_root = "/content/Datasets/kaggle_knee_osteoarthritis"
    checkpoint_dir = "/content/drive/MyDrive/Models/densenet201_checkpoints"
    img_size = 224
    batch_size = 16
    seed = 42
    
    # Imbalance Handling Options
    use_balanced_sampler = True       # True/False (WeightedRandomSampler for train split)
    use_minority_aug = True           # True/False (Stronger augmentations for Grade 3 & 4)
    
    # Training Stage & Freezing Options
    # Options: 
    #   "3-stage": Stage 1 (FC Warm-up), Stage 2 (Coarse tuning), Stage 3 (Fine-tuning)
    #   "2-stage": Stage 1 (FC Warm-up), Stage 2 (Full training)
    #   "standard": Standard full fine-tuning directly (no freezing, standard CE/CORN loss)
    training_pipeline = "3-stage" 
    
    # Phase Epochs
    stage1_epochs = 5
    stage2_epochs = 25
    stage3_epochs = 15
    total_epochs_standard = 20        # Only used if training_pipeline is "standard"
    
    # Learning Rates
    lr_warmup = 1e-3
    lr_coarse_head = 1e-4
    lr_coarse_backbone = 1e-5         # Lower learning rate for CNN backbone in Stage 2/3
    lr_finetune = 1e-5
    lr_standard = 1e-4                # Learning rate if pipeline is "standard"
    
    weight_decay = 1e-4
    
    # Loss Function Choices
    # Options: "ce" (Cross-Entropy), "corn" (Conditional Ordinal), "coral" (Rank Ordinal), "focal_corn" (Focal CORN)
    loss_stage1 = "corn"
    loss_stage2 = "corn"
    loss_stage3 = "focal_corn"
    loss_standard = "ce"              # Used only if training_pipeline is "standard"
    
    # Learning Rate Schedulers
    # Options: "cosine" (CosineAnnealingLR), "step" (StepLR), "plateau" (ReduceLROnPlateau), "none"
    scheduler_stage2 = "none"
    scheduler_stage3 = "cosine"
    scheduler_standard = "cosine"

def log_config(config):
    print("="*65)
    print(" ACTIVE TRAINING CONFIGURATION LOG")
    print("="*65)
    attrs = [attr for attr in dir(config) if not attr.startswith('__') and not callable(getattr(config, attr))]
    for attr in attrs:
        print(f"{attr:<25} : {getattr(config, attr)}")
    print("="*65)

# Log configurations
log_config(TrainingConfig)

# Set global alias variables for compatibility with downstream cells
DATASET_ROOT_PATH = TrainingConfig.dataset_root
CHECKPOINT_SAVE_DIR = TrainingConfig.checkpoint_dir
BATCH_SIZE = TrainingConfig.batch_size
IMG_SIZE = TrainingConfig.img_size

# Set random seed for reproducibility
torch.manual_seed(TrainingConfig.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(TrainingConfig.seed)
np.random.seed(TrainingConfig.seed)
import random
random.seed(TrainingConfig.seed)

os.makedirs(CHECKPOINT_SAVE_DIR, exist_ok=True)"""

cell_3_code = """class SquarePadOpenCV(object):
    \"\"\"Pads a rectangular image to a square.\"\"\"
    def __call__(self, image):
        h, w = image.shape[:2]
        max_wh = max(h, w)
        pad_top = (max_wh - h) // 2
        pad_bottom = max_wh - h - pad_top
        pad_left = (max_wh - w) // 2
        pad_right = max_wh - w - pad_left
        
        padded_image = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right, 
            borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        return padded_image

class OpenCVCLAHE(object):
    \"\"\"Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) using OpenCV.\"\"\"
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(img_lab)
        clahe_l_channel = clahe.apply(l_channel)
        merged_lab_image = cv2.merge((clahe_l_channel, a_channel, b_channel))
        return cv2.cvtColor(merged_lab_image, cv2.COLOR_LAB2RGB)

def get_transforms(img_size=224):
    \"\"\"Returns training, validation/test, and stronger minority class transforms.\"\"\"
    train_transform = transforms.Compose([
        SquarePadOpenCV(),
        OpenCVCLAHE(),
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        SquarePadOpenCV(),
        OpenCVCLAHE(),
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Stronger augmentations for minority classes (Grade 3 & 4) to prevent overfitting during oversampling
    minority_train_transform = transforms.Compose([
        SquarePadOpenCV(),
        OpenCVCLAHE(),
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform, minority_train_transform

def remove_duplicate_images(image_paths: list, labels: list, exclude_hashes: set = None):
    \"\"\"Removes duplicate images using MD5 hashing.\"\"\"
    total_found = len(image_paths)
    unique_paths, unique_labels, unique_hashes = [], [], set()
    internal_dup_count, leakage_count = 0, 0
    
    for path, label in zip(image_paths, labels):
        hash_md5 = hashlib.md5()
        try:
            with open(path, \"rb\") as f:
                for chunk in iter(lambda: f.read(4096), b\"\"): 
                    hash_md5.update(chunk)
            h = hash_md5.hexdigest()
        except Exception as e:
            print(f\"Warning: Could not read image {path}: {e}\")
            continue
            
        if exclude_hashes and h in exclude_hashes:
            leakage_count += 1
            continue
        if h in unique_hashes:
            internal_dup_count += 1
            continue
            
        unique_hashes.add(h)
        unique_paths.append(path)
        unique_labels.append(label)
        
    print(f\"\\n--- Deduplication: Files found: {total_found} | Unique kept: {len(unique_paths)} | Dupes removed: {internal_dup_count} | Cross-split leaks: {leakage_count}\")
    return unique_paths, unique_labels, unique_hashes"""

cell_4_code = """class KaggleKneeOsteoarthritisDataset(Dataset):
    \"\"\"Dataset class for loading Kaggle Knee OA dataset splits.\"\"\"
    def __init__(self, root: str, split_dir: str, transform=None, exclude_hashes: set = None, minority_transform=None):
        self.root = root
        self.transform = transform
        self.minority_transform = minority_transform
        self.exclude_hashes = exclude_hashes
        raw_paths, raw_labels = [], []
        split_path = os.path.join(root, split_dir)
        
        if not os.path.isdir(split_path): 
            raise FileNotFoundError(f\"Split directory not found: {split_path}\")
            
        class_names = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d)) and d.isdigit()])
        print(f\"Loading '{split_dir}' split from: {split_path}\")
        
        for class_name in class_names:
            class_dir = os.path.join(split_path, class_name)
            label = int(class_name)
            valid_extensions = ('.png', '.jpg', '.jpeg')
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(valid_extensions)]
            for file_name in image_files:
                raw_paths.append(os.path.join(class_dir, file_name))
                raw_labels.append(label)
                
        self.image_paths, self.labels, self.image_hashes = remove_duplicate_images(
            raw_paths, raw_labels, exclude_hashes=self.exclude_hashes
        )

    def load_image_from_path(self, image_path: str) -> np.ndarray:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None: 
            raise IOError(f\"Could not read image: {image_path}\")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int):
        image = self.load_image_from_path(self.image_paths[idx])
        label = self.labels[idx]
        
        # Apply stronger transform if it is a minority class (Grade 3 & 4)
        if self.minority_transform and label in [3, 4]:
            image = self.minority_transform(image)
        elif self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self) -> int: 
        return len(self.image_paths)"""

cell_5_code = """# Create transforms
train_transform, val_transform, minority_train_transform = get_transforms(img_size=IMG_SIZE)

# Determine minority transform based on configuration
minor_transform = minority_train_transform if TrainingConfig.use_minority_aug else None

# Load training dataset
train_dataset = KaggleKneeOsteoarthritisDataset(
    root=DATASET_ROOT_PATH, split_dir="train", transform=train_transform, minority_transform=minor_transform
)
train_hashes = set(train_dataset.image_hashes)

# Load validation dataset
val_split_dir = "val" if os.path.isdir(os.path.join(DATASET_ROOT_PATH, "val")) else "test"
val_dataset = KaggleKneeOsteoarthritisDataset(
    root=DATASET_ROOT_PATH, split_dir=val_split_dir, transform=val_transform, exclude_hashes=train_hashes
)

# Imbalance handling: Calculate Class-Aware WeightedRandomSampler for training split
from torch.utils.data import WeightedRandomSampler

# Count samples of each class
class_counts = Counter(train_dataset.labels)
print(f"Training class distribution: {dict(sorted(class_counts.items()))}")

# Create training loader based on config
if TrainingConfig.use_balanced_sampler:
    print("Using WeightedRandomSampler for class balance.")
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_dataset.labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
else:
    print("Using standard shuffled DataLoader.")
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Data loaders ready. Train batches: {len(train_loader)} | Validation batches: {len(val_loader)}")"""

cell_6_code = """def label_to_levels(label, num_classes, dtype=torch.float32):
    # Converts class label index to binary level vector
    batch_size = label.size(0)
    levels = torch.zeros(batch_size, num_classes - 1, dtype=dtype, device=label.device)
    for i in range(batch_size):
        levels[i, :label[i]] = 1.0
    return levels

def coral_loss(logits, y_train, num_classes=5):
    levels = label_to_levels(y_train, num_classes)
    loss = F.binary_cross_entropy_with_logits(logits, levels)
    return loss

def coral_label_from_logits(logits):
    probs = torch.sigmoid(logits)
    predicted = (probs > 0.5).sum(dim=1)
    return predicted

def corn_loss(logits, y_train, num_classes=5):
    loss = 0.0
    num_tasks = num_classes - 1
    for k in range(num_tasks):
        mask = y_train >= k
        if not mask.any():
            continue
        logits_k = logits[mask, k]
        targets_k = (y_train[mask] > k).float()
        loss += F.binary_cross_entropy_with_logits(logits_k, targets_k)
    return loss / num_tasks

def focal_corn_loss(logits, y_train, num_classes=5, gamma=2.0, alpha=0.25):
    loss = 0.0
    num_tasks = num_classes - 1
    for k in range(num_tasks):
        mask = y_train >= k
        if not mask.any():
            continue
        logits_k = logits[mask, k]
        targets_k = (y_train[mask] > k).float()
        
        bce = F.binary_cross_entropy_with_logits(logits_k, targets_k, reduction='none')
        p = torch.sigmoid(logits_k)
        
        # Calculate focal modulation weight
        p_t = p * targets_k + (1 - p) * (1 - targets_k)
        focal_weight = alpha * (1 - p_t) ** gamma
        
        loss += (focal_weight * bce).mean()
    return loss / num_tasks

def corn_probas(logits):
    cond_probas = torch.sigmoid(logits)
    batch_size = logits.size(0)
    num_classes = logits.size(1) + 1
    probas = torch.zeros(batch_size, num_classes, device=logits.device)
    cumprod = torch.cumprod(cond_probas, dim=1)
    probas[:, 0] = 1.0 - cond_probas[:, 0]
    for i in range(1, num_classes - 1):
        probas[:, i] = cumprod[:, i - 1] * (1.0 - cond_probas[:, i])
    probas[:, -1] = cumprod[:, -1]
    return probas

def corn_label_from_logits(logits):
    probas = corn_probas(logits)
    return torch.argmax(probas, dim=1)

class DenseNet201Model(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = True, loss_type: str = "corn"):
        super(DenseNet201Model, self).__init__()
        
        # Set head dimensions. CE has num_classes (5), Ordinal (CORN/CORAL) has num_classes - 1 (4)
        out_features = num_classes if loss_type == "ce" else num_classes - 1
        
        # Load standard DenseNet-201 model
        self.model = timm.create_model('densenet201', pretrained=pretrained, num_classes=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def freeze_backbone(self):
        print("Freezing DenseNet-201 backbone features.")
        for param in self.model.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        print("Unfreezing all DenseNet-201 parameters.")
        for param in self.model.parameters():
            param.requires_grad = True

    def fit(self, epoch, data_loader, optimizer, loss_type, device):
        self.to(device)
        self.train()
        running_loss, total, correct = 0.0, 0, 0
        
        criterion = get_loss_criterion(loss_type, num_classes=5)
        predict_fn = get_prediction_helper(loss_type)
        
        progress_bar = tqdm.tqdm(data_loader, desc=f"Epoch {epoch+1} [TRAIN]")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = self(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            
            predicted = predict_fn(outputs)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.0 * correct / total:.2f}%"
            })
            
        return running_loss / total, 100.0 * correct / total

    def evaluate(self, epoch, data_loader, loss_type, device, description="VALIDATE"):
        self.to(device)
        self.eval()
        running_loss, total, correct = 0.0, 0, 0
        all_preds, all_labels, all_probas = [], [], []
        
        criterion = get_loss_criterion(loss_type, num_classes=5)
        predict_fn = get_prediction_helper(loss_type)
        
        # Select probas resolver: CE uses Softmax, CORN uses chain rule, CORAL uses Sigmoid
        if loss_type == "ce":
            probas_fn = lambda x: F.softmax(x, dim=1)
        elif loss_type in ["corn", "focal_corn"]:
            probas_fn = corn_probas
        elif loss_type == "coral":
            probas_fn = torch.sigmoid
            
        progress_bar = tqdm.tqdm(data_loader, desc=f"Epoch {epoch+1} [{description}]" if epoch is not None else description)
        with torch.no_grad():
            for images, labels in progress_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                
                probas = probas_fn(outputs)
                predicted = predict_fn(outputs)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probas.extend(probas.cpu().numpy())
                
        # Convert to numpy arrays for sklearn metrics
        all_labels_arr = np.array(all_labels)
        all_probas_arr = np.array(all_probas)
        
        # Pad probas array to 5 classes if it has 4 (for CORAL/CORN) during OVR calculations
        if all_probas_arr.shape[1] == 4:
            # We can calculate probas for CORN
            if loss_type in ["corn", "focal_corn"]:
                pass # Already 5 classes returned by corn_probas!
            elif loss_type == "coral":
                # Convert CORAL logits to 5 classes probabilities
                temp = np.zeros((all_probas_arr.shape[0], 5))
                temp[:, 0] = 1.0 - all_probas_arr[:, 0]
                for idx in range(1, 4):
                    temp[:, idx] = all_probas_arr[:, idx - 1] - all_probas_arr[:, idx]
                temp[:, 4] = all_probas_arr[:, 3]
                all_probas_arr = np.clip(temp, 0.0, 1.0)
                
        report = classification_report(
            all_labels_arr, all_preds, 
            target_names=[str(i) for i in range(5)], 
            zero_division=0
        )
        
        # Calculate Cohen's Quadratic Weighted Kappa
        from torchmetrics.classification import CohenKappa
        kappa_metric = CohenKappa(task="multiclass", num_classes=5, weights="quadratic")
        kappa_score = kappa_metric(torch.tensor(all_preds), torch.tensor(all_labels)).item()
        
        # Calculate AUC and AP (One-vs-Rest Macro)
        from sklearn.metrics import roc_auc_score, average_precision_score
        try:
            auc_score = roc_auc_score(all_labels_arr, all_probas_arr, multi_class='ovr', average='macro')
        except Exception as e:
            auc_score = 0.0
            print(f"ROC AUC computation failed: {e}")
            
        try:
            y_one_hot = np.eye(5)[all_labels_arr]
            ap_score = average_precision_score(y_one_hot, all_probas_arr, average='macro')
        except Exception as e:
            ap_score = 0.0
            print(f"Average Precision (AP) computation failed: {e}")
            
        print(f"\\nQuadratic Weighted Kappa (QWK): {kappa_score:.4f}")
        print(f"ROC AUC (OVR Macro): {auc_score:.4f}")
        print(f"Average Precision (AP Macro): {ap_score:.4f}")
        
        metrics = {
            "loss": running_loss / total,
            "acc": 100.0 * correct / total,
            "qwk": kappa_score,
            "auc": auc_score,
            "ap": ap_score,
            "probas": all_probas_arr,
            "report": report
        }
        return metrics"""

cell_7_code = """# Instantiate DenseNet-201 matching the active pipeline initial configuration
initial_loss = TrainingConfig.loss_stage1 if TrainingConfig.training_pipeline in ["2-stage", "3-stage"] else TrainingConfig.loss_standard
model = DenseNet201Model(num_classes=5, pretrained=TrainingConfig.pretrained, loss_type=initial_loss)

# Define stage checkpoints paths
stage2_best_path = os.path.join(CHECKPOINT_SAVE_DIR, "stage2_best_model.pth")
stage3_best_path = os.path.join(CHECKPOINT_SAVE_DIR, "best_model.pth")
last_model_path = os.path.join(CHECKPOINT_SAVE_DIR, "last_model.pth")

# (Ignore Cross Entropy loss as we use CORN loss inside fit/evaluate)
criterion = None

print(f"DenseNet-201 Model initialized with {initial_loss.upper()} output configuration!")"""

cell_8_code = """best_val_qwk = -1.0
best_val_qwk_stage2 = -1.0
history = []

# Define loss helper functions inside the cell to ensure they are available
def get_loss_criterion(loss_type, num_classes=5):
    if loss_type == "ce":
        return nn.CrossEntropyLoss()
    elif loss_type == "corn":
        return lambda logits, targets: corn_loss(logits, targets, num_classes)
    elif loss_type == "coral":
        return lambda logits, targets: coral_loss(logits, targets, num_classes)
    elif loss_type == "focal_corn":
        return lambda logits, targets: focal_corn_loss(logits, targets, num_classes)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def get_prediction_helper(loss_type):
    if loss_type == "ce":
        return lambda logits: torch.argmax(logits, dim=1)
    elif loss_type in ["corn", "focal_corn"]:
        return corn_label_from_logits
    elif loss_type == "coral":
        return coral_label_from_logits
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def get_scheduler(scheduler_type, optimizer, epochs):
    if scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    elif scheduler_type == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs*0.33), gamma=0.1)
    elif scheduler_type == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    else:
        return None

# Execute pipeline based on configuration
pipeline = TrainingConfig.training_pipeline
print(f"Executing Training Pipeline: {pipeline.upper()}")

if pipeline == "3-stage":
    # -------------------------------------------------------------------------
    # STAGE 1: Warm-up FC
    # -------------------------------------------------------------------------
    print("\\n=== STAGE 1: WARM-UP FC ===")
    model.freeze_backbone()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=TrainingConfig.lr_warmup, weight_decay=TrainingConfig.weight_decay)
    
    stage1_epochs = TrainingConfig.stage1_epochs
    loss_type = TrainingConfig.loss_stage1
    
    for epoch in range(stage1_epochs):
        train_loss, train_acc = model.fit(epoch, train_loader, optimizer, loss_type, device)
        val_metrics = model.evaluate(epoch, val_loader, loss_type, device, description="VALIDATE")
        val_loss, val_acc, val_qwk = val_metrics["loss"], val_metrics["acc"], val_metrics["qwk"]
        
        history.append({
            "stage": "Stage 1", "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "qwk": val_qwk, "auc": val_metrics["auc"], "ap": val_metrics["ap"]
        })

    # -------------------------------------------------------------------------
    # STAGE 2: Coarse-tuning
    # -------------------------------------------------------------------------
    print("\\n=== STAGE 2: COARSE-TUNING ===")
    model.unfreeze_backbone()
    
    # Configure discriminative learning rates
    optimizer = optim.AdamW([
        {'params': model.model.features.parameters(), 'lr': TrainingConfig.lr_coarse_backbone},
        {'params': model.model.classifier.parameters(), 'lr': TrainingConfig.lr_coarse_head}
    ], weight_decay=TrainingConfig.weight_decay)
    
    stage2_epochs = TrainingConfig.stage2_epochs
    loss_type = TrainingConfig.loss_stage2
    scheduler = get_scheduler(TrainingConfig.scheduler_stage2, optimizer, stage2_epochs)
    
    for epoch in range(stage1_epochs, stage1_epochs + stage2_epochs):
        train_loss, train_acc = model.fit(epoch, train_loader, optimizer, loss_type, device)
        val_metrics = model.evaluate(epoch, val_loader, loss_type, device, description="VALIDATE")
        val_loss, val_acc, val_qwk = val_metrics["loss"], val_metrics["acc"], val_metrics["qwk"]
        
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        history.append({
            "stage": "Stage 2", "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "qwk": val_qwk, "auc": val_metrics["auc"], "ap": val_metrics["ap"]
        })
        
        if val_qwk > best_val_qwk_stage2:
            best_val_qwk_stage2 = val_qwk
            torch.save({'model_state_dict': model.state_dict()}, stage2_best_path)
            print(f"--> Saved best Stage 2 model with QWK: {best_val_qwk_stage2:.4f}")

    # -------------------------------------------------------------------------
    # STAGE 3: Fine-tuning
    # -------------------------------------------------------------------------
    print("\\n=== STAGE 3: FINE-TUNING ===")
    if os.path.exists(stage2_best_path):
        print("Loading best Stage 2 weights...")
        model.load_state_dict(torch.load(stage2_best_path)['model_state_dict'])
    
    # Switch to Original DataLoader (No Sampler)
    train_loader_orig = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    train_dataset.minority_transform = None
    
    optimizer = optim.AdamW(model.parameters(), lr=TrainingConfig.lr_finetune, weight_decay=10*TrainingConfig.weight_decay)
    
    stage3_epochs = TrainingConfig.stage3_epochs
    loss_type = TrainingConfig.loss_stage3
    scheduler = get_scheduler(TrainingConfig.scheduler_stage3, optimizer, stage3_epochs)
    
    for epoch in range(stage1_epochs + stage2_epochs, stage1_epochs + stage2_epochs + stage3_epochs):
        train_loss, train_acc = model.fit(epoch, train_loader_orig, optimizer, loss_type, device)
        val_metrics = model.evaluate(epoch, val_loader, loss_type, device, description="VALIDATE")
        val_loss, val_acc, val_qwk = val_metrics["loss"], val_metrics["acc"], val_metrics["qwk"]
        
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
                
        history.append({
            "stage": "Stage 3", "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "qwk": val_qwk, "auc": val_metrics["auc"], "ap": val_metrics["ap"]
        })
        
        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            torch.save({'model_state_dict': model.state_dict()}, stage3_best_path)
            print(f"--> Saved best Stage 3 final model with QWK: {best_val_qwk:.4f}")
            
        torch.save({'model_state_dict': model.state_dict()}, last_model_path)

elif pipeline == "2-stage":
    # -------------------------------------------------------------------------
    # STAGE 1: Warm-up FC
    # -------------------------------------------------------------------------
    print("\\n=== STAGE 1: WARM-UP FC ===")
    model.freeze_backbone()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=TrainingConfig.lr_warmup, weight_decay=TrainingConfig.weight_decay)
    
    stage1_epochs = TrainingConfig.stage1_epochs
    loss_type = TrainingConfig.loss_stage1
    
    for epoch in range(stage1_epochs):
        train_loss, train_acc = model.fit(epoch, train_loader, optimizer, loss_type, device)
        val_metrics = model.evaluate(epoch, val_loader, loss_type, device, description="VALIDATE")
        val_loss, val_acc, val_qwk = val_metrics["loss"], val_metrics["acc"], val_metrics["qwk"]
        
        history.append({
            "stage": "Stage 1", "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "qwk": val_qwk, "auc": val_metrics["auc"], "ap": val_metrics["ap"]
        })

    # -------------------------------------------------------------------------
    # STAGE 2: Full training
    # -------------------------------------------------------------------------
    print("\\n=== STAGE 2: FULL FINE-TUNING ===")
    model.unfreeze_backbone()
    optimizer = optim.AdamW(model.parameters(), lr=TrainingConfig.lr_coarse_head, weight_decay=TrainingConfig.weight_decay)
    
    stage2_epochs = TrainingConfig.stage2_epochs
    loss_type = TrainingConfig.loss_stage2
    scheduler = get_scheduler(TrainingConfig.scheduler_stage2, optimizer, stage2_epochs)
    
    for epoch in range(stage1_epochs, stage1_epochs + stage2_epochs):
        train_loss, train_acc = model.fit(epoch, train_loader, optimizer, loss_type, device)
        val_metrics = model.evaluate(epoch, val_loader, loss_type, device, description="VALIDATE")
        val_loss, val_acc, val_qwk = val_metrics["loss"], val_metrics["acc"], val_metrics["qwk"]
        
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
                
        history.append({
            "stage": "Stage 2", "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "qwk": val_qwk, "auc": val_metrics["auc"], "ap": val_metrics["ap"]
        })
        
        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            torch.save({'model_state_dict': model.state_dict()}, stage3_best_path)
            print(f"--> Saved best model checkpoint with QWK: {best_val_qwk:.4f}")
            
        torch.save({'model_state_dict': model.state_dict()}, last_model_path)

else:
    # -------------------------------------------------------------------------
    # STANDARD TRAINING (1 Stage)
    # -------------------------------------------------------------------------
    print("\\n=== STANDARD FINE-TUNING ===")
    model.unfreeze_backbone()
    optimizer = optim.AdamW(model.parameters(), lr=TrainingConfig.lr_standard, weight_decay=TrainingConfig.weight_decay)
    
    total_epochs = TrainingConfig.total_epochs_standard
    loss_type = TrainingConfig.loss_standard
    scheduler = get_scheduler(TrainingConfig.scheduler_standard, optimizer, total_epochs)
    
    for epoch in range(total_epochs):
        train_loss, train_acc = model.fit(epoch, train_loader, optimizer, loss_type, device)
        val_metrics = model.evaluate(epoch, val_loader, loss_type, device, description="VALIDATE")
        val_loss, val_acc, val_qwk = val_metrics["loss"], val_metrics["acc"], val_metrics["qwk"]
        
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
                
        history.append({
            "stage": "Standard", "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "qwk": val_qwk, "auc": val_metrics["auc"], "ap": val_metrics["ap"]
        })
        
        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            torch.save({'model_state_dict': model.state_dict()}, stage3_best_path)
            print(f"--> Saved best model checkpoint with QWK: {best_val_qwk:.4f}")
            
        torch.save({'model_state_dict': model.state_dict()}, last_model_path)

# Print final training history log summary table
print("\\n" + "="*95)
print("TRAINING HISTORY LOG SUMMARY")
print("="*95)
print(f"{'Stage':<9} | {'Epoch':<5} | {'Train Loss':<10} | {'Train Acc':<9} | {'Val Loss':<8} | {'Val Acc':<7} | {'QWK':<6} | {'ROC AUC':<7} | {'AP':<6}")
print("-"*95)
for h in history:
    print(f"{h['stage']:<9} | {h['epoch']:<5} | {h['train_loss']:<10.4f} | {h['train_acc']:<8.2f}% | {h['val_loss']:<8.4f} | {h['val_acc']:<6.2f}% | {h['qwk']:<6.4f} | {h['auc']:<7.4f} | {h['ap']:<6.4f}")
print("="*95)"""

cell_10_code = """# Load the test dataset split
test_dataset = KaggleKneeOsteoarthritisDataset(
    root=DATASET_ROOT_PATH, split_dir="test", transform=val_transform, exclude_hashes=train_hashes
)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Loaded test dataset containing {len(test_dataset)} images.")

# Load the best model weights
if os.path.exists(best_model_path):
    print("Loading best model checkpoint for testing...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    print("Best model checkpoint not found. Testing with current weights.")

# Run evaluation on test dataset and collect raw outputs for plotting
model.to(device)
model.eval()
all_preds, all_labels, all_probas = [], [], []

loss_type = TrainingConfig.loss_stage3 if TrainingConfig.training_pipeline == "3-stage" else (TrainingConfig.loss_stage2 if TrainingConfig.training_pipeline == "2-stage" else TrainingConfig.loss_standard)
predict_fn = get_prediction_helper(loss_type)

if loss_type == "ce":
    probas_fn = lambda x: F.softmax(x, dim=1)
elif loss_type in ["corn", "focal_corn"]:
    probas_fn = corn_probas
elif loss_type == "coral":
    probas_fn = torch.sigmoid

with torch.no_grad():
    for images, labels in tqdm.tqdm(test_loader, desc="TEST EVALUATION"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probas = probas_fn(outputs)
        predicted = predict_fn(outputs)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probas.extend(probas.cpu().numpy())

# Convert to numpy arrays
y_true = np.array(all_labels)
y_pred = np.array(all_preds)
y_probas = np.array(all_probas)

# Pad probas array to 5 classes if it has 4 (for CORAL/CORN) during OVR calculations
if y_probas.shape[1] == 4:
    if loss_type in ["corn", "focal_corn"]:
        pass # Already 5 classes returned by corn_probas!
    elif loss_type == "coral":
        temp = np.zeros((y_probas.shape[0], 5))
        temp[:, 0] = 1.0 - y_probas[:, 0]
        for idx in range(1, 4):
            temp[:, idx] = y_probas[:, idx - 1] - y_probas[:, idx]
        temp[:, 4] = y_probas[:, 3]
        y_probas = np.clip(temp, 0.0, 1.0)

# 1. Compute basic metrics
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve, auc
from torchmetrics.classification import CohenKappa
import torch

test_acc = accuracy_score(y_true, y_pred)
kappa_metric = CohenKappa(task="multiclass", num_classes=5, weights="quadratic")
test_qwk = kappa_metric(torch.tensor(y_pred), torch.tensor(y_true)).item()
test_auc = roc_auc_score(y_true, y_probas, multi_class='ovr', average='macro')
y_one_hot = np.eye(5)[y_true]
test_ap = average_precision_score(y_one_hot, y_probas, average='macro')

# 2. Compute 95% Confidence Intervals using Bootstrapping
print("\\nComputing 95% Confidence Intervals via bootstrapping (200 iterations)...")
boot_acc, boot_qwk, boot_auc, boot_ap = [], [], [], []
rng = np.random.default_rng(42)
for _ in range(200):
    indices = rng.choice(len(y_true), size=len(y_true), replace=True)
    if len(np.unique(y_true[indices])) < 5:
        continue
    y_true_b = y_true[indices]
    y_pred_b = y_pred[indices]
    y_probas_b = y_probas[indices]
    
    boot_acc.append(accuracy_score(y_true_b, y_pred_b))
    boot_qwk.append(kappa_metric(torch.tensor(y_pred_b), torch.tensor(y_true_b)).item())
    try:
        boot_auc.append(roc_auc_score(y_true_b, y_probas_b, multi_class='ovr', average='macro'))
    except:
        pass
    try:
        y_one_hot_b = np.eye(5)[y_true_b]
        boot_ap.append(average_precision_score(y_one_hot_b, y_probas_b, average='macro'))
    except:
        pass

def get_ci(data):
    sorted_data = np.sort(data)
    low = sorted_data[int(0.025 * len(sorted_data))]
    high = sorted_data[int(0.975 * len(sorted_data))]
    return low, high

acc_ci = get_ci(boot_acc)
qwk_ci = get_ci(boot_qwk)
auc_ci = get_ci(boot_auc)
ap_ci = get_ci(boot_ap)

print("\\n" + "="*50)
print("=== FINAL TEST METRICS WITH 95% CONFIDENCE INTERVALS ===")
print("="*50)
print(f"Accuracy: {test_acc:.4f} (95% CI: {acc_ci[0]:.4f} - {acc_ci[1]:.4f})")
print(f"QWK Score: {test_qwk:.4f} (95% CI: {qwk_ci[0]:.4f} - {qwk_ci[1]:.4f})")
print(f"ROC AUC: {test_auc:.4f} (95% CI: {auc_ci[0]:.4f} - {auc_ci[1]:.4f})")
print(f"Average Precision (AP): {test_ap:.4f} (95% CI: {ap_ci[0]:.4f} - {ap_ci[1]:.4f})")
print("="*50)

print("\\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=[str(i) for i in range(5)], zero_division=0))

# 3. Plotting Diagrams
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    has_sns = True
except ImportError:
    has_sns = False

plt.figure(figsize=(24, 7))

# Plot 1: Confusion Matrix
plt.subplot(1, 3, 1)
cm = confusion_matrix(y_true, y_pred)
if has_sns:
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[str(i) for i in range(5)], yticklabels=[str(i) for i in range(5)], cbar=False, annot_kws={"size": 14})
else:
    plt.imshow(cm, cmap='Blues')
    for i in range(5):
        for j in range(5):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=14)
plt.title("Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Grade", fontsize=12)
plt.ylabel("True Grade", fontsize=12)

# Plot 2: ROC Curves (OVR)
plt.subplot(1, 3, 2)
for i in range(5):
    fpr, tpr, _ = roc_curve(y_one_hot[:, i], y_probas[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Grade {i} (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.title("One-vs-Rest ROC Curves", fontsize=16)
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.legend(loc="lower right")

# Plot 3: Precision-Recall Curves (OVR)
plt.subplot(1, 3, 3)
for i in range(5):
    precision, recall, _ = precision_recall_curve(y_one_hot[:, i], y_probas[:, i])
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f"Grade {i} (AP = {pr_auc:.4f})")
plt.title("One-vs-Rest Precision-Recall Curves", fontsize=16)
plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.legend(loc="lower left")

plt.tight_layout()
plt.show()"""

cell_12_code = """class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None
        
        # Register hooks
        self.hook_forward = self.target_layer.register_forward_hook(self.save_features)
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.hook_backward = self.target_layer.register_full_backward_hook(self.save_gradients)
        else:
            self.hook_backward = self.target_layer.register_backward_hook(self.save_gradients)
        
    def save_features(self, module, input, output):
        self.features = output
        
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)
        
        loss_type = TrainingConfig.loss_stage3 if TrainingConfig.training_pipeline == "3-stage" else (TrainingConfig.loss_stage2 if TrainingConfig.training_pipeline == "2-stage" else TrainingConfig.loss_standard)
        predict_fn = get_prediction_helper(loss_type)
        
        if class_idx is None:
            class_idx = predict_fn(output).item()
            
        self.model.zero_grad()
        # For CORN / CORAL task indexing
        if loss_type == "ce":
            loss = output[0, class_idx]
        else:
            task_idx = min(max(class_idx - 1, 0), output.size(1) - 1)
            loss = output[0, task_idx]
            
        loss.backward()
        
        # Pool gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Apply weights to features
        cam = torch.sum(weights * self.features, dim=1).squeeze(0)
        
        # Apply ReLU to retain positive influence features
        cam = F.relu(cam)
        cam = cam.cpu().detach().numpy()
        
        if cam.max() > 0:
            cam = cam / cam.max()
            
        cam = cv2.resize(cam, (x.shape[2], x.shape[3]))
        return cam, class_idx
        
    def remove_hooks(self):
        self.hook_forward.remove()
        self.hook_backward.remove()

def show_gradcam(image_path, model, target_layer, transform):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
        
    # Read and preprocess image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process original image dimensions for display
    pad = SquarePadOpenCV()
    clahe = OpenCVCLAHE()
    img_processed = clahe(pad(img_rgb))
    img_resized = cv2.resize(img_processed, (IMG_SIZE, IMG_SIZE))
    
    # Tensor transform
    tensor = transform(img_rgb).unsqueeze(0).to(device)
    
    # Run Grad-CAM
    gradcam = GradCAM(model, target_layer)
    cam, class_idx = gradcam(tensor)
    gradcam.remove_hooks()
    
    # Create colormap overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    alpha = 0.4
    overlay = cv2.addWeighted(img_resized, 1 - alpha, heatmap, alpha, 0)
    
    # Plot side-by-side
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Knee X-ray (Processed)")
    plt.imshow(img_resized)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Grad-CAM Heatmap (Predicted Grade: {class_idx})")
    plt.imshow(overlay)
    plt.axis('off')
    
    plt.show()

# Specify the last convolutional layer of DenseNet-201 for target
# In timm's densenet201, this corresponds to model.model.features.norm5
target_layer = model.model.features.norm5

# Example usage (uncomment and replace with your image path):
# example_image_path = "/content/Datasets/kaggle_knee_osteoarthritis/test/4/9003887R.png"
# show_gradcam(example_image_path, model, target_layer, val_transform)"""

# Load the notebook
with open("notebooks/dense_net_201.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Helper function to convert multi-line string into notebook list of strings format
def to_source_list(code_str):
    lines = code_str.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]

# Modify source code of target cells
nb["cells"][1]["source"] = to_source_list(cell_1_code)
nb["cells"][2]["source"] = to_source_list(cell_2_code)
nb["cells"][3]["source"] = to_source_list(cell_3_code)
nb["cells"][4]["source"] = to_source_list(cell_4_code)
nb["cells"][5]["source"] = to_source_list(cell_5_code)
nb["cells"][6]["source"] = to_source_list(cell_6_code)
nb["cells"][7]["source"] = to_source_list(cell_7_code)
nb["cells"][8]["source"] = to_source_list(cell_8_code)
nb["cells"][10]["source"] = to_source_list(cell_10_code)
nb["cells"][12]["source"] = to_source_list(cell_12_code)

# Remove cell output history to keep the notebook clean and ready for user run
for i in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]:
    if "outputs" in nb["cells"][i]:
        nb["cells"][i]["outputs"] = []
    if "execution_count" in nb["cells"][i]:
        nb["cells"][i]["execution_count"] = None

# Save the updated notebook
with open("notebooks/dense_net_201.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully with complete dynamic TrainingConfig framework!")
