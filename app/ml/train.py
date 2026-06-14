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
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
ROOT = os.path.join(BASE_DIR, "data", "Knee X-ray Images")
TRAIN_DATASET_DIR = "MedicalExpert-I"
VALIDATE_DATASET_DIR = "MedicalExpert-II"
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "model_weights")
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. MAIN TRAINING PIPELINE
# ==========================================
def main():
    print(f"[Device] Using device: {DEVICE}")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # 1. Initialize Datasets & Dataloaders
    train_transform, val_transform = get_transforms(img_size=224)
    
    train_dataset = KneeXRayDataset(
        root=ROOT, 
        train_dataset_dir=TRAIN_DATASET_DIR, 
        validate_dataset_dir=VALIDATE_DATASET_DIR, 
        transform=train_transform, 
        train=True
    )
    val_dataset = KneeXRayDataset(
        root=ROOT, 
        train_dataset_dir=TRAIN_DATASET_DIR, 
        validate_dataset_dir=VALIDATE_DATASET_DIR, 
        transform=val_transform, 
        train=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # 2. Initialize Model Wrapper
    model = EfficientNetB0Model(num_classes=5, pretrained=True)
    model = model.to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    
    last_weights_path = os.path.join(MODEL_SAVE_DIR, 'last_weights.pth')
    best_weights_path = os.path.join(MODEL_SAVE_DIR, 'best_weights.pth')
    
    epochs = 10
    for idx in range(epochs):
        print(f"\n--- Epoch {idx+1}/{epochs} ---")
        
        # Training
        train_loss, train_acc = model.fit(
            epoch=idx,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE
        )
        print(f"  -> Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        
        # Evaluating
        eval_loss, eval_acc, report = model.evaluate(
            epoch=idx,
            data_loader=val_loader,
            criterion=criterion,
            device=DEVICE
        )
        print(f"  -> Val Loss: {eval_loss:.4f} | Val Acc: {eval_acc:.2f}%")
        print("Val Classification Report:\n", report)
        
        # Save last weights
        torch.save(model.state_dict(), last_weights_path)
        print(f"Saved: {last_weights_path}")
        
        # Save best weights
        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(model.state_dict(), best_weights_path)
            print(f"New Best Model Saved: {best_weights_path} (Acc: {best_acc:.2f}%)")

if __name__ == '__main__':
    main()
