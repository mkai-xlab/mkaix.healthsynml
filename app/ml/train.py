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

    # Load training dataset
    train_dataset = DatasetClass(root=str(root_path), split_dir="train", transform=train_transform)
    train_hashes = set(train_dataset.image_hashes)
    
    val_dir_path = os.path.join(root_path, "val")
    val_split_dir = "val" if os.path.isdir(val_dir_path) else "test"
    
    # Load validation dataset, excluding training hashes to prevent data leakage
    val_dataset = DatasetClass(root=str(root_path), split_dir=val_split_dir, transform=val_transform, exclude_hashes=train_hashes)
    
    # If the validation dataset is empty after removing training duplicates,
    # perform a dynamic random split on the unique training images.
    if len(val_dataset) == 0:
        print("\n[WARNING] Validation dataset is empty after removing training duplicates (leakage)!")
        print("This indicates that the train and validation folders are identical or highly overlapping.")
        print("Performing a dynamic 80/20 train/validation split on unique images to prevent leakage.")
        
        import random
        all_paths = list(train_dataset.image_paths)
        all_labels = list(train_dataset.labels)
        
        # Shuffle with a fixed seed for reproducibility
        combined = list(zip(all_paths, all_labels))
        random.seed(42)
        random.shuffle(combined)
        
        split_idx = int(len(combined) * 0.8)
        train_pairs = combined[:split_idx]
        val_pairs = combined[split_idx:]
        
        # Update train_dataset in-place
        train_dataset.image_paths = [p for p, _ in train_pairs]
        train_dataset.labels = [l for _, l in train_pairs]
        train_dataset.image_hashes = set()
        
        # Update val_dataset in-place
        val_dataset.image_paths = [p for p, _ in val_pairs]
        val_dataset.labels = [l for _, l in val_pairs]
        val_dataset.image_hashes = set()
        
        # Print post-split statistics
        from collections import Counter
        print(f"\n--- Post-Split Dataset Statistics (Dynamic 80/20) ---")
        print(f"  - Training images: {len(train_dataset)}")
        print(f"  - Validation images: {len(val_dataset)}")
        
        train_counts = Counter(train_dataset.labels)
        val_counts = Counter(val_dataset.labels)
        
        print("  - Training class distribution:")
        for label_idx in sorted(train_counts.keys()):
            class_name = train_dataset.categories[label_idx] if hasattr(train_dataset, 'categories') and label_idx < len(train_dataset.categories) else f"Class {label_idx}"
            print(f"    * {class_name}: {train_counts[label_idx]} images")
            
        print("  - Validation class distribution:")
        for label_idx in sorted(val_counts.keys()):
            class_name = val_dataset.categories[label_idx] if hasattr(val_dataset, 'categories') and label_idx < len(val_dataset.categories) else f"Class {label_idx}"
            print(f"    * {class_name}: {val_counts[label_idx]} images")
        print("------------------------------------------------------\n")
        
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
    early_stopping_patience: int = 10,
    ordinal_type: str = "none"
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

    num_classes = 4 if ordinal_type == "threshold" else 5
    model = get_model(model_name, num_classes=num_classes, pretrained=True)
    
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
    print(f"  - Ordinal Type: {ordinal_type}")
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
        if model_name == "efficientnet_b0":
            print("    - Applying Discriminative Fine-Tuning (3 groups) for EfficientNet-B0")
            early_backbone_params = []
            late_backbone_params = []
            classifier_params = []
            for n, p in model.named_parameters():
                if 'classifier' in n:
                    classifier_params.append(p)
                elif 'features' in n:
                    parts = n.split('.')
                    try:
                        features_idx = parts.index('features')
                        block_idx = int(parts[features_idx + 1])
                        # Blocks 0-3 are early, blocks 4-8 are late (similar to transfer learning division)
                        if block_idx < 4:
                            early_backbone_params.append(p)
                        else:
                            late_backbone_params.append(p)
                    except (ValueError, IndexError):
                        early_backbone_params.append(p)
                else:
                    early_backbone_params.append(p)
            optimizer = optim.AdamW([
                {'params': early_backbone_params, 'lr': lr * 0.01},
                {'params': late_backbone_params, 'lr': lr * 0.1},
                {'params': classifier_params, 'lr': lr}
            ], weight_decay=1e-2)
            print(f"      - Discriminative LR: Early Backbone LR = {lr*0.01:.7f}, Late Backbone LR = {lr*0.1:.6f}, Classifier LR = {lr:.6f}")
        else:
            # Fallback to standard 2-group split for other models
            backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
            classifier_params = [p for n, p in model.named_parameters() if 'classifier' in n]
            optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': lr * 0.1},
                {'params': classifier_params, 'lr': lr}
            ], weight_decay=1e-2)
            print(f"      - Differential LR: Backbone LR = {lr*0.1:.6f}, Classifier LR = {lr:.6f}")
    
    if ordinal_type == "threshold":
        criterion = "ordinal_threshold"
        print("  - Loss Function: Binary Cross Entropy with Logits Loss (Frank-Hall Threshold)")
    elif ordinal_type == "expected_value":
        if use_focal_loss:
            criterion = "expected_value_focal_loss"
            print("  - Loss Function: Sigmoid Focal Loss + Expected Value Regularization")
        else:
            criterion = "expected_value_cross_entropy"
            print("  - Loss Function: Cross Entropy Loss + Expected Value Regularization")
    else:
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
        try:
            checkpoint = torch.load(last_model_path_local, map_location=device)
            model.load_state_dict(checkpoint["model"])
            if "optimizer" in checkpoint and optimizer is not None: 
                optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint and scheduler is not None:
                try: 
                    scheduler.load_state_dict(checkpoint["scheduler"])
                except Exception: 
                    print("Warning: Could not load scheduler state dict.")
            current_epoch = checkpoint.get("epoch", 0) + 1
            if early_stopper:
                if "val_loss_min" in checkpoint:
                    early_stopper.val_loss_min = checkpoint["val_loss_min"]
                    early_stopper.best_score = -checkpoint["val_loss_min"]
                elif "val_loss" in checkpoint:
                    early_stopper.val_loss_min = checkpoint["val_loss"]
                    early_stopper.best_score = -checkpoint["val_loss"]
                if "early_stop_counter" in checkpoint:
                    early_stopper.counter = checkpoint["early_stop_counter"]
            if "rng_state" in checkpoint:
                torch.set_rng_state(checkpoint["rng_state"].cpu())
            if "cuda_rng_state" in checkpoint and checkpoint["cuda_rng_state"] is not None and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["cuda_rng_state"]])
                except Exception:
                    pass
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor): state[k] = v.to(device)
            print(f"Successfully resumed training from epoch {current_epoch}.")
        except Exception as e:
            print(f"Warning: Could not load checkpoint due to corruption or mismatch ({e}). Training will start from scratch.")
            current_epoch = 0
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

        # Save last model checkpoint locally (using atomic write to prevent corruption)
        checkpoint = {
            "model": model.state_dict(), 
            "optimizer": optimizer.state_dict(), 
            "scheduler": scheduler.state_dict(), 
            "epoch": epoch, 
            "val_loss": val_loss,
            "val_loss_min": early_stopper.val_loss_min if early_stopper else val_loss,
            "early_stop_counter": early_stopper.counter if early_stopper else 0,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
        tmp_path = f"{last_model_path_local}.tmp"
        torch.save(checkpoint, tmp_path)
        if os.path.exists(tmp_path):
            os.replace(tmp_path, last_model_path_local)

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
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training and validation.")
    parser.add_argument("--img-size", type=int, default=380, help="Image size for input images.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--fine-tune", action="store_true", help="Unfreeze and fine-tune the backbone with a lower learning rate.")
    parser.add_argument("--focal-loss", action="store_true", help="Use Sigmoid Focal Loss instead of CrossEntropyLoss.")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Patience for early stopping. Set to 0 to disable.")
    parser.add_argument("--ordinal-type", type=str, default="none", choices=["none", "expected_value", "threshold"], help="Type of ordinal classification strategy.")
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
        early_stopping_patience=args.early_stopping_patience,
        ordinal_type=args.ordinal_type
    )
