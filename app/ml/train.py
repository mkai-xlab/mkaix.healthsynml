import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary

from app.ml.dataset import KneeXRayDataset, get_transforms
from app.ml.model_registry import get_model
from app.ml.models.base_model import save_model_dict, load_model_dict
from app.utils.file_utils import find_and_remove_duplicates
from app.utils.s3_utils import s3_object_exists
from app.core.config import settings

def setup_device():
    """Sets up the device for training."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    return device

def prepare_dataloaders(root_path, train_dir, val_dir, batch_size, img_size):
    """Pre-processes data and creates DataLoader for training and validation."""
    print("\n--- Cleaning up duplicate images ---")
    full_train_path = os.path.join(root_path, train_dir)
    full_val_path = os.path.join(root_path, val_dir)
    find_and_remove_duplicates(full_train_path)
    find_and_remove_duplicates(full_val_path)
    print("--- Clean up complete ---\n")

    train_transform, val_transform = get_transforms(img_size=img_size)

    train_dataset = KneeXRayDataset(
        root=root_path,
        train_dataset_dir=train_dir,
        validate_dataset_dir=val_dir,
        train=True,
        transform=train_transform
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = KneeXRayDataset(
        root=root_path,
        train_dataset_dir=train_dir,
        validate_dataset_dir=val_dir,
        train=False,
        transform=val_transform
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def run_training(model_name: str, epochs: int = 100, batch_size: int = 32, img_size: int = 224, lr: float = 1e-3):
    """
    Main function to run the training and validation pipeline.
    """
    device = setup_device()

    # Prepare DataLoaders
    dataset_root_path = os.path.join("data", "Knee X-ray Images")
    train_loader, val_loader = prepare_dataloaders(
        root_path=dataset_root_path,
        train_dir="MedicalExpert-I",
        val_dir="MedicalExpert-II",
        batch_size=batch_size,
        img_size=img_size
    )

    # Initialize model, optimizer, and criterion
    model = get_model(model_name, num_classes=5, pretrained=True)
    model.freeze_backbone()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # Initialize scheduler
    total_steps = epochs * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # Load checkpoint if exists
    best_acc = 0.0
    current_epoch = 0
    model_sates_path = "models_states"
    model_save_dir = os.path.join(model_sates_path, model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    
    last_model_key = os.path.join(model_save_dir, "last_model.pth").replace("\\", "/")
    if s3_object_exists(settings.AWS_S3_MODELS_BUCKET, last_model_key):
        print(f"Loading checkpoint for model '{model_name}' from S3.")
        current_epoch, best_acc = load_model_dict(
            model=model, 
            path=last_model_key, 
            optimizer=optimizer,
            device=device,
        )
        # Ensure optimizer state is on the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    else:
        print(f"No checkpoint found for model '{model_name}'. Starting from scratch.")

    print(summary(model, (1, 3, img_size, img_size)))

    # Training loop
    for epoch in range(current_epoch, epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        train_loss, train_acc = model.fit(
            epoch=epoch, 
            data_loader=train_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            device=device, 
            scheduler=scheduler
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        val_loss, val_acc, _ = model.evaluate(
            epoch=epoch, 
            data_loader=val_loader, 
            criterion=criterion, 
            device=device
        )
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save checkpoints
        if val_acc > best_acc:
            best_acc = val_acc
            best_save_path = os.path.join(model_save_dir, "best_model.pth")
            save_model_dict(model=model, path=best_save_path, epoc=epoch, optimizer=optimizer, bess_acc=best_acc)
            print(f"Saved new best model with Val Acc: {best_acc:.2f}% to {best_save_path}")
        
        last_save_path = os.path.join(model_save_dir, "last_model.pth")
        save_model_dict(model=model, path=last_save_path, epoc=epoch, optimizer=optimizer, bess_acc=best_acc)

if __name__ == '__main__':
    # This allows the script to be run directly.
    # You can use argparse here to select the model and other parameters.
    import argparse
    parser = argparse.ArgumentParser(description="Run model training.")
    parser.add_argument("--model", type=str, default="mobilenet_v2", help="Name of the model to train.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    args = parser.parse_args()
    
    run_training(model_name=args.model, epochs=args.epochs)
