import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from app.ml.models.efficientnet_b0_model import EfficientNetB0Model
from app.ml.dataset import KneeXRayDataset
from app.ml.dataset import get_transforms
from app.ml.models.base_model import  save_model_dict
from app.utils.file_utils import find_and_remove_duplicates

if __name__ == "__main__":
    # overview pytorch
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))

    # get current device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    batch_size = 32
    epochs = 50
    model = EfficientNetB0Model(num_classes=5, pretrained=True)
    model.freeze_backbone()
    train_transform, val_transform = get_transforms(img_size=224)
    dataset_root_path = os.path.join("data", "Knee X-ray Images")
    dataset_train_path = "MedicalExpert-I"
    dataset_val_path = "MedicalExpert-II"

    # Pre-process: Clean up duplicate images
    print("\n--- Cleaning up duplicate images ---")
    full_train_path = os.path.join(dataset_root_path, dataset_train_path)
    full_val_path = os.path.join(dataset_root_path, dataset_val_path)
    find_and_remove_duplicates(full_train_path)
    find_and_remove_duplicates(full_val_path)
    print("--- Clean up complete ---\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_knee_xray_dataset = KneeXRayDataset(root=dataset_root_path,
                                              train_dataset_dir=dataset_train_path,
                                              validate_dataset_dir=dataset_val_path,
                                              train=True, transform=train_transform)
    train_knee_xray_dataloader = DataLoader(dataset=train_knee_xray_dataset, batch_size= batch_size, shuffle=True)

    val_knee_xray_dataset = KneeXRayDataset(root=dataset_root_path,
                                            train_dataset_dir=dataset_train_path,
                                            validate_dataset_dir=dataset_val_path,
                                            train=False, transform= val_transform)
    val_knee_xray_dataloader = DataLoader(dataset=val_knee_xray_dataset, batch_size= batch_size, shuffle=False)

    best_acc = 0.0
    model_name = "efficientnet_b0"
    model_save_dir = os.path.join("models_states", model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        train_loss, train_acc = model.fit(epoch=epoch, data_loader=train_knee_xray_dataloader, criterion=criterion, optimizer=optimizer, device=device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        val_loss, val_acc, report = model.evaluate(epoch=epoch, data_loader=val_knee_xray_dataloader, criterion=criterion, device=device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        # print(report) # Optional: Uncomment to print full classification report each epoch

        # Save last model at every epoch
        last_save_path = os.path.join(model_save_dir, "last_model.pth")
        save_model_dict(model=model, path=last_save_path, epoc=epoch, optimizer=optimizer, bess_acc=best_acc)

        # Check and save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_save_path = os.path.join(model_save_dir, "best_model.pth")
            save_model_dict(model=model, path=best_save_path, epoc=epoch, optimizer=optimizer, bess_acc=best_acc)
            print(f"Saved new best model with Val Acc: {best_acc:.2f}% to {best_save_path}")