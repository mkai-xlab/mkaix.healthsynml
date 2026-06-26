import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
from pathlib import Path

# Import helpers and registries
from app.ml.dataset import get_transforms
from app.ml.model_registry import get_model
from app.ml.dataset_registry import get_dataset_info
from app.ml.models.base_model import save_model_dict, load_model_dict
from app.utils.early_stopping import EarlyStopping
from app.core.config import settings

def setup_device():
    """Sets up the device for training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    return device

def prepare_dataloaders(dataset_name, root_path, batch_size, img_size):
    """
    Prepares DataLoaders based on the selected dataset name from the registry.
    """
    train_transform, val_transform = get_transforms(img_size=img_size)
    
    DatasetClass = get_dataset_info(dataset_name)["class"]

    train_dataset = DatasetClass(root=str(root_path), split_dir="train", transform=train_transform)
    
    val_dir_path = os.path.join(root_path, "val")
    val_split_dir = "val" if os.path.isdir(val_dir_path) else "test"
    val_dataset = DatasetClass(root=str(root_path), split_dir=val_split_dir, transform=val_transform)
    
    print(f"Loaded datasets: {len(train_dataset)} for training, {len(val_dataset)} for validation.")

    num_workers = 4 if torch.cuda.is_available() else 0
    pin_memory = True if torch.cuda.is_available() else False

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader

def run_training(
    model_name: str, 
    dataset_name: str,
    epochs: int = 100, 
    batch_size: int = 32, 
    img_size: int = 224, 
    lr: float = 1e-4,
    fine_tune: bool = False,
    use_focal_loss: bool = False,
    early_stopping_patience: int = 10
):
    """
    Main function to run the training and validation pipeline.
    """
    device = setup_device()

    dataset_info = get_dataset_info(dataset_name)
    dataset_root_path = dataset_info["default_path"]
    
    train_loader, val_loader = prepare_dataloaders(
        dataset_name=dataset_name,
        root_path=dataset_root_path,
        batch_size=batch_size,
        img_size=img_size
    )

    model = get_model(model_name, num_classes=5, pretrained=True)
    
    # --- Configuration Summary ---
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION SUMMARY")
    print("="*50)
    print(f"  - Model: {model_name}")
    print(f"  - Dataset: {dataset_name}")
    print(f"  - Image Size: {img_size}x{img_size}")
    print(f"  - Max Epochs: {epochs}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Initial Learning Rate: {lr}")
    if early_stopping_patience > 0:
        print(f"  - Early Stopping: Enabled (patience={early_stopping_patience})")
    else:
        print("  - Early Stopping: Disabled")


    if not fine_tune:
        model.freeze_backbone()
        freezing_strategy = "Backbone frozen"
        if model_name == "efficientnet_b0":
            freezing_strategy += " (Partial: blocks 4-7 & classifier are trainable)"
        print(f"  - Freezing Strategy: {freezing_strategy}")
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    else:
        print("  - Freezing Strategy: Fine-tuning all layers")
        backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
        classifier_params = [p for n, p in model.named_parameters() if 'classifier' in n]
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': lr * 0.1},
            {'params': classifier_params, 'lr': lr}
        ], weight_decay=1e-2)
        print(f"    - Differential LR: Backbone LR = {lr*0.1:.6f}, Classifier LR = {lr:.6f}")
    
    if use_focal_loss:
        criterion = "focal_loss"
        print("  - Loss Function: Sigmoid Focal Loss (gamma=2.0, alpha=0.25)")
    else:
        criterion = nn.CrossEntropyLoss()
        print("  - Loss Function: Cross Entropy Loss")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    print(f"  - Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)")
    print("="*50 + "\n")
    
    # --- Checkpoint and Early Stopping Setup ---
    current_epoch = 0
    project_root = Path(__file__).parent.parent.parent
    checkpoint_dir_relative = "checkpoints"
    model_save_dir_absolute = project_root / checkpoint_dir_relative / model_name
    os.makedirs(model_save_dir_absolute, exist_ok=True)
    
    last_model_path_local = model_save_dir_absolute / "last_model.pth"
    best_model_path_local = model_save_dir_absolute / "best_model.pth"

    early_stopper = None
    if early_stopping_patience > 0:
        early_stopper = EarlyStopping(patience=early_stopping_patience, verbose=True, path=best_model_path_local)

    if os.path.exists(last_model_path_local):
        print(f"Loading local checkpoint for model '{model_name}' from: {last_model_path_local}")
        checkpoint = torch.load(last_model_path_local, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint and optimizer is not None: optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and scheduler is not None:
            try: scheduler.load_state_dict(checkpoint["scheduler"])
            except Exception: print("Warning: Could not load scheduler state dict.")
        current_epoch = checkpoint.get("epoch", 0) + 1
        if early_stopper and "val_loss" in checkpoint:
            early_stopper.val_loss_min = checkpoint["val_loss"]
            early_stopper.best_score = -checkpoint["val_loss"]
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor): state[k] = v.to(device)
    else:
        print(f"No checkpoint found locally for model '{model_name}'. Starting from scratch.")

    print("\n--- Model Summary ---")
    print(summary(model, (1, 3, img_size, img_size), verbose=0))
    print("--- End Model Summary ---\n")

    # --- Training Loop ---
    for epoch in range(current_epoch, epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        lrs = [group['lr'] for group in optimizer.param_groups]
        print(f"Current learning rate(s): {lrs}")
        
        train_loss, train_acc = model.fit(epoch=epoch, data_loader=train_loader, criterion=criterion, optimizer=optimizer, device=device, scheduler=scheduler)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        val_loss, val_acc, _ = model.evaluate(epoch=epoch, data_loader=val_loader, criterion=criterion, device=device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau): scheduler.step(val_loss)

        # Save last model checkpoint locally
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "val_loss": val_loss}
        torch.save(checkpoint, last_model_path_local)

        # Early stopping check (also saves the best model internally if loss improves)
        if early_stopper:
            if early_stopper(val_loss, model, optimizer, scheduler, epoch):
                print("Early stopping triggered!")
                break

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run model training.")
    parser.add_argument("--model", type=str, default="mobilenet_v2", help="Name of the model to train.")
    parser.add_argument("--dataset-name", type=str, default="kaggle", choices=["local", "kaggle", "mendeley"], help="Name of the dataset to use.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training and validation.")
    parser.add_argument("--img-size", type=int, default=224, help="Image size for input images.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--fine-tune", action="store_true", help="Unfreeze and fine-tune the backbone with a lower learning rate.")
    parser.add_argument("--focal-loss", action="store_true", help="Use Sigmoid Focal Loss instead of CrossEntropyLoss.")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Patience for early stopping. Set to 0 to disable.")
    args = parser.parse_args()
    
    run_training(
        model_name=args.model,
        dataset_name=args.dataset_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        lr=args.lr,
        fine_tune=args.fine_tune,
        use_focal_loss=args.focal_loss,
        early_stopping_patience=args.early_stopping_patience
    )
