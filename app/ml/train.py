import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add root directory to path to support app namespace imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.ml.dataset import KneeXRayDataset, get_transforms
from app.ml.models.efficientnet_b0_model import EfficientNetB0Model

# ==========================================
# 1. TRAINING CONFIGURATIONS
# ==========================================
ROOT = "/content/drive/MyDrive/Digital_Knee_X_ray_Images"
TRAIN_DATASET_DIR = "MedicalExpert-I"
VALIDATE_DATASET_DIR = "MedicalExpert-II"
MODEL_SAVE_DIR = "/content/drive/MyDrive/AI/models/EffiicentNetB0"
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. CORE TRAIN & VALIDATION FUNCTIONS
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc=" Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / total, 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=" Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / total, 100. * correct / total

# ==========================================
# 3. MAIN TRAINING PIPELINE
# ==========================================
def main():
    print(f"[Device] Using device: {DEVICE}")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # 1. Initialize Datasets & Dataloaders
    train_transform, val_transform = get_transforms(img_size=224)
    
    train_dataset = KneeXRayDataset(root=ROOT, dataset_dir=TRAIN_DATASET_DIR, transform=train_transform)
    val_dataset = KneeXRayDataset(root=ROOT, dataset_dir=VALIDATE_DATASET_DIR, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # 2. Initialize Model Wrapper
    model_wrapper = EfficientNetB0Model(num_classes=5, pretrained=True)
    model = model_wrapper.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    
    # ==========================================
    # STAGE 1: FEATURE EXTRACTION (Train classifier head only)
    # ==========================================
    print("\n=== STAGE 1: HUẤN LUYỆN LỚP CLASSIFIER ===")
    # Freeze backbone parameters
    for param in model.model.parameters():
        param.requires_grad = False
    # Unfreeze classifier parameters
    for param in model.model.classifier.parameters():
        param.requires_grad = True
        
    optimizer_stage1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    epochs_stage1 = 5
    best_val_acc = 0.0
    stage1_checkpoint_path = os.path.join(MODEL_SAVE_DIR, "best_efficientnet_b0_stage1.pth")
    
    for epoch in range(epochs_stage1):
        print(f"\nEpoch {epoch+1}/{epochs_stage1} (Stage 1)")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_stage1, DEVICE)
        print(f"  -> Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        print(f"  -> Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), stage1_checkpoint_path)
            print(f"  [Save] Saved Stage 1 Checkpoint: {best_val_acc:.2f}%")
            
    # ==========================================
    # STAGE 2: FINE-TUNING (Train whole network with small LR)
    # ==========================================
    print("\n=== STAGE 2: FINE-TUNING TOÀN BỘ MÔ HÌNH ===")
    # Load best weights from Stage 1
    if os.path.exists(stage1_checkpoint_path):
        model.load_state_dict(torch.load(stage1_checkpoint_path, weights_only=True))
        print(f"Loaded weights from Stage 1: {stage1_checkpoint_path}")
        
    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer_stage2 = optim.Adam(model.parameters(), lr=1e-5)
    
    epochs_stage2 = 15
    best_val_acc_finetune = best_val_acc
    finetune_checkpoint_path = os.path.join(MODEL_SAVE_DIR, "best_efficientnet_b0_finetuned.pth")
    
    for epoch in range(epochs_stage2):
        print(f"\nEpoch {epoch+1}/{epochs_stage2} (Stage 2 - Fine-Tuning)")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_stage2, DEVICE)
        print(f"  -> Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        print(f"  -> Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc_finetune:
            best_val_acc_finetune = val_acc
            torch.save(model.state_dict(), finetune_checkpoint_path)
            print(f"  [Save] Saved Fine-Tuned Checkpoint: {best_val_acc_finetune:.2f}%")
            
    print(f"\nTraining Complete! Best accuracy: {best_val_acc_finetune:.2f}%")

if __name__ == '__main__':
    main()
