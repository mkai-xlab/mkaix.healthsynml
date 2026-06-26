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
from app.utils.file_utils import find_and_remove_duplicates
from app.utils.s3_utils import s3_object_exists
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
    lr: float = 1e-3,
    fine_tune: bool = False,
    use_focal_loss: bool = False
):
    """
    Main function to run the training and validation pipeline.
    """
    device = setup_device()

    dataset_info = get_dataset_info(dataset_name)
    dataset_root_path = dataset_info["default_path"]
    
    # This logic was incorrect and has been removed.
    # The path from the registry is now used directly.
    # if dataset_name == "kaggle":
    #     dataset_root_path = dataset_root_path / f"kneeKL{img_size}"

    print(f"Using dataset: '{dataset_name}' from root path: {dataset_root_path}")
    
    train_loader, val_loader = prepare_dataloaders(
        dataset_name=dataset_name,
        root_path=dataset_root_path,
        batch_size=batch_size,
        img_size=img_size
    )

    model = get_model(model_name, num_classes=5, pretrained=True)
    
    if not fine_tune:
        model.freeze_backbone()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    else:
        backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
        classifier_params = [p for n, p in model.named_parameters() if 'classifier' in n]
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': lr * 0.1},
            {'params': classifier_params, 'lr': lr}
        ], weight_decay=1e-2)
    
    if use_focal_loss:
        print("Using Sigmoid Focal Loss.")
        criterion = "focal_loss"
    else:
        print("Using standard Cross Entropy Loss.")
        criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=2, min_lr=1e-6)

    best_acc = 0.0
    current_epoch = 0
    
    project_root = Path(__file__).parent.parent.parent
    checkpoint_dir_relative = "checkpoints"
    model_save_dir_absolute = project_root / checkpoint_dir_relative / model_name
    os.makedirs(model_save_dir_absolute, exist_ok=True)
    
    last_model_path_local = model_save_dir_absolute / "last_model.pth"
    best_model_path_local = model_save_dir_absolute / "best_model.pth"

    if os.path.exists(last_model_path_local):
        print(f"Loading local checkpoint for model '{model_name}' from: {last_model_path_local}")
        checkpoint = torch.load(last_model_path_local, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint and optimizer is not None: optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and scheduler is not None:
            try: scheduler.load_state_dict(checkpoint["scheduler"])
            except Exception: print("Warning: Could not load scheduler state dict.")
        current_epoch = checkpoint.get("epoch", 0) + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor): state[k] = v.to(device)
    else:
        print(f"No checkpoint found locally for model '{model_name}'. Starting from scratch.")

    print(summary(model, (1, 3, img_size, img_size)))

    for epoch in range(current_epoch, epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        lrs = [group['lr'] for group in optimizer.param_groups]
        print(f"Current learning rate(s): {lrs}")
        
        train_loss, train_acc = model.fit(epoch=epoch, data_loader=train_loader, criterion=criterion, optimizer=optimizer, device=device, scheduler=scheduler)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        val_loss, val_acc, _ = model.evaluate(epoch=epoch, data_loader=val_loader, criterion=criterion, device=device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau): scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "best_acc": best_acc}
            torch.save(checkpoint, best_model_path_local)
            print(f"Saved new best model locally to: {best_model_path_local} with Val Acc: {best_acc:.2f}%")
        
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "best_acc": best_acc}
        torch.save(checkpoint, last_model_path_local)
        print(f"Saved last model checkpoint locally to: {last_model_path_local}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run model training.")
    parser.add_argument("--model", type=str, default="mobilenet_v2", help="Name of the model to train.")
    parser.add_argument("--dataset-name", type=str, default="kaggle", 
                        choices=["local", "kaggle", "mendeley"], 
                        help="Name of the dataset to use.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training and validation.")
    parser.add_argument("--img-size", type=int, default=224, help="Image size for input images.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--fine-tune", action="store_true", help="Unfreeze and fine-tune the backbone with a lower learning rate.")
    parser.add_argument("--focal-loss", action="store_true", help="Use Sigmoid Focal Loss instead of CrossEntropyLoss.")
    args = parser.parse_args()
    
    run_training(
        model_name=args.model,
        dataset_name=args.dataset_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        lr=args.lr,
        fine_tune=args.fine_tune,
        use_focal_loss=args.focal_loss
    )
