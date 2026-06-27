import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sklearn.metrics import classification_report
from torchvision.ops import sigmoid_focal_loss
import os

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

            # --- Loss & Prediction Calculation ---
            if criterion == "ordinal_threshold":
                num_classes_minus_1 = output.shape[1]
                targets = (labels.unsqueeze(1) > torch.arange(num_classes_minus_1, device=device)).float()
                loss = F.binary_cross_entropy_with_logits(output, targets)
                predicted = (torch.sigmoid(output) > 0.5).sum(dim=1)
            elif criterion == "expected_value_cross_entropy" or criterion == "expected_value_focal_loss":
                if criterion == "expected_value_focal_loss":
                    targets = F.one_hot(labels, num_classes=output.shape[1]).float()
                    base_loss = sigmoid_focal_loss(output, targets, alpha=0.25, gamma=2.0, reduction='mean')
                else:
                    base_loss = F.cross_entropy(output, labels)
                
                probs = F.softmax(output, dim=1)
                class_indices = torch.arange(output.shape[1], dtype=torch.float32, device=device)
                expected_y = torch.sum(probs * class_indices, dim=1)
                ord_loss = F.smooth_l1_loss(expected_y, labels.float())
                loss = base_loss + 2.0 * ord_loss
                _, predicted = torch.max(output.data, 1)
            elif criterion == "focal_loss":
                targets = F.one_hot(labels, num_classes=output.shape[1]).float()
                loss = sigmoid_focal_loss(output, targets, alpha=0.25, gamma=2.0, reduction='mean')
                _, predicted = torch.max(output.data, 1)
            else:
                loss = criterion(output, labels)
                _, predicted = torch.max(output.data, 1)

            loss.backward()
            optimizer.step()
            
            if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            running_loss += loss.item() * labels.size(0)
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

                # --- Loss & Prediction Calculation ---
                if criterion == "ordinal_threshold":
                    num_classes_minus_1 = output.shape[1]
                    targets = (labels.unsqueeze(1) > torch.arange(num_classes_minus_1, device=device)).float()
                    loss = F.binary_cross_entropy_with_logits(output, targets)
                    predicted = (torch.sigmoid(output) > 0.5).sum(dim=1)
                elif criterion == "expected_value_cross_entropy" or criterion == "expected_value_focal_loss":
                    if criterion == "expected_value_focal_loss":
                        targets = F.one_hot(labels, num_classes=output.shape[1]).float()
                        base_loss = sigmoid_focal_loss(output, targets, alpha=0.25, gamma=2.0, reduction='mean')
                    else:
                        base_loss = F.cross_entropy(output, labels)
                    
                    probs = F.softmax(output, dim=1)
                    class_indices = torch.arange(output.shape[1], dtype=torch.float32, device=device)
                    expected_y = torch.sum(probs * class_indices, dim=1)
                    ord_loss = F.smooth_l1_loss(expected_y, labels.float())
                    loss = base_loss + 2.0 * ord_loss
                    _, predicted = torch.max(output.data, 1)
                elif criterion == "focal_loss":
                    targets = F.one_hot(labels, num_classes=output.shape[1]).float()
                    loss = sigmoid_focal_loss(output, targets, alpha=0.25, gamma=2.0, reduction='mean')
                    _, predicted = torch.max(output.data, 1)
                else:
                    loss = criterion(output, labels)
                    _, predicted = torch.max(output.data, 1)

                running_loss += loss.item() * labels.size(0)
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
