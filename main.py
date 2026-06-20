import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary


from app.ml.models.mobilenet_v2_model import MobileNetV2Model
from app.ml.dataset import KneeXRayDataset
from app.ml.dataset import get_transforms
from app.ml.models.base_model import save_model_dict, load_model_dict
from app.utils.s3_utils import s3_object_exists
from app.core.config import settings
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
    epochs = 100
    model = MobileNetV2Model(num_classes=5, pretrained=True)
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
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    train_knee_xray_dataset = KneeXRayDataset(root=dataset_root_path,
                                              train_dataset_dir=dataset_train_path,
                                              validate_dataset_dir=dataset_val_path,
                                              train=True, transform=train_transform)
    train_knee_xray_dataloader = DataLoader(dataset=train_knee_xray_dataset, batch_size=batch_size, shuffle=True)

    val_knee_xray_dataset = KneeXRayDataset(root=dataset_root_path,
                                            train_dataset_dir=dataset_train_path,
                                            validate_dataset_dir=dataset_val_path,
                                            train=False, transform=val_transform)
    val_knee_xray_dataloader = DataLoader(dataset=val_knee_xray_dataset, batch_size=batch_size, shuffle=False)

    best_acc = 0.0
    current_epoch = 0
    model_name = "mobilenet_v2"
    model_sates_path = "models_states"
    model_save_dir = os.path.join(model_sates_path, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    total_steps = epochs * len(train_knee_xray_dataloader)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)


    # load old model weight, current epoch, optimizer from S3
    last_model_key = os.path.join(model_save_dir, "last_model.pth").replace("\\", "/")
    if s3_object_exists(settings.AWS_S3_MODELS_BUCKET, last_model_key):
        current_epoch, best_acc = load_model_dict(
            model=model, 
            path=last_model_key, 
            optimizer=optimizer,
            device=device,
        )

        # load to current device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    print(summary(model, (1, 3, 224, 224)))


    for epoch in range(current_epoch, epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        train_loss, train_acc = model.fit(epoch=epoch, data_loader=train_knee_xray_dataloader, criterion=criterion, optimizer=optimizer, device=device, scheduler=scheduler)
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