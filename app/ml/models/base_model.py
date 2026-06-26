import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sklearn.metrics import classification_report
from torchvision.ops import sigmoid_focal_loss

from app.core.config import settings
from app.utils.s3_utils import upload_to_s3, download_from_s3

class BaseModel(nn.Module):
    """
    Abstract Base Model Wrapper defining the contract for all neural networks
    in the Knee Osteoarthritis classification system.
    """
    def __init__(self):
        super(BaseModel, self).__init__()
        
    def forward(self, x):
        pass

    def freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def fit(self, epoch, data_loader, optimizer, criterion, device, scheduler=None):
        self.to(device)
        self.train()

        running_loss = 0.0
        total = 0
        correct = 0
        progress_bar = tqdm.tqdm(data_loader, desc=f"Epoch {epoch+1} [TRAIN]")
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = self(images)

            # --- Loss Calculation ---
            if criterion == "focal_loss":
                # Convert labels to one-hot format for focal loss
                targets = F.one_hot(labels, num_classes=output.shape[1]).float()
                loss = sigmoid_focal_loss(output, targets, alpha=0.25, gamma=2.0, reduction='mean')
            else:
                # Standard loss function (e.g., CrossEntropyLoss)
                loss = criterion(output, labels)

            loss.backward()
            optimizer.step()
            
            if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.2f}%")

        return running_loss / total, 100 * correct / total

    def evaluate(self, epoch, data_loader, criterion, device):
        self.to(device)
        self.eval()
        
        running_loss = 0.0
        total = 0
        correct = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm.tqdm(data_loader, desc=f"Epoch {epoch+1} [VALIDATE]")

        with torch.no_grad():
            for images, labels in progress_bar:
                images, labels = images.to(device), labels.to(device)
                output = self(images)

                if criterion == "focal_loss":
                    targets = F.one_hot(labels, num_classes=output.shape[1]).float()
                    loss = sigmoid_focal_loss(output, targets, alpha=0.25, gamma=2.0, reduction='mean')
                else:
                    loss = criterion(output, labels)

                running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.2f}%")

        report = classification_report(y_true=all_labels, y_pred=all_predictions, zero_division=0)
        return running_loss / total, 100 * correct / total, report

# ... (save_model_dict and load_model_dict need to be updated to handle scheduler)
def save_model_dict(model: nn.Module, path, epoc: int, optimizer: torch.optim.Optimizer, scheduler, bess_acc):
    # This function might need to handle S3 or local saving based on a config
    # For now, assuming local saving for simplicity based on recent user changes
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoc,
        "best_acc": bess_acc
    }
    torch.save(checkpoint, path)

def load_model_dict(model: nn.Module, path, optimizer: torch.optim.Optimizer = None, scheduler=None, device: torch.device = torch.device("cpu")):
    # This function might need to handle S3 or local loading
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")
        
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler"])
        except Exception as e:
            print(f"Warning: Could not load scheduler state dict: {e}")
            
    return checkpoint.get("epoch", 0), checkpoint.get("best_acc", 0.0)
