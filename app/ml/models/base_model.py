import io
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import classification_report
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
        for param in self.parameters():
            param.requires_grad = False

        if self.use_timm:
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        else:
            for param in self.model.classifier.parameters():
                param.requires_grad = True

    def load_weights(self, path: str, device: torch.device):
        """
        Loads weight checkpoint from disk onto the target device.
        """
        raise NotImplementedError("Subclasses must implement load_weights method")

    def fit(self, epoch, data_loader, optimizer, criterion, device):
        self.to(device)
        self.train()

        running_loss = 0.0
        total = 0
        correct = 0
        progress_bar = tqdm.tqdm(data_loader)
        num_iters = len(data_loader)

        for idx, (images, labels) in enumerate(progress_bar):
            # load to device
            images = images.to(device)
            labels = labels.to(device)

            # forward
            output = self(images)
            loss = criterion(output, labels)

            progress_bar.set_description("Epoch {}: Iteration {}/{}. Loss {:.3f}".format(epoch+1, idx+1, num_iters, loss))
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == labels).sum().item()

        return running_loss / total, 100 * correct / total

    def evaluate(self, epoch, data_loader, criterion, device):
        all_predictions = []
        all_labels = []
        self.to(device)
        self.eval()
        running_loss = 0.0
        total = 0
        correct = 0
        num_iters = len(data_loader)
        progress_bar = tqdm.tqdm(data_loader)
        for idx, (images, labels) in enumerate(progress_bar):
            # load to device
            images = images.to(device)
            labels = labels.to(device)

            # forward
            output = self(images)
            loss = criterion(output, labels)
            progress_bar.set_description("Epoch {}: Iteration {}/{}. Loss {:.3f}".format(epoch+1, idx+1, num_iters, loss))

            # calc loss
            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == labels).sum().item()

            # report
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predicted.cpu().tolist())

        report = classification_report(y_true=all_labels, y_pred=all_predictions)

        return running_loss / total, 100 * correct / total, report

    def predict(self, x):
        """
        Runs forward pass and returns raw logits/probabilities.
        """
        return self(x)

# save state dict
def save_model_dict(model: nn.Module, path, epoc: int, optimizer: torch.optim.Optimizer, bess_acc: int):
    s3_key = str(path).replace("\\", "/")
    bucket = settings.AWS_S3_MODELS_BUCKET
    
    print(f"Saving checkpoint to S3: s3://{bucket}/{s3_key}")
    entrypoint = dict()
    entrypoint["model"] = model.state_dict()
    entrypoint["optimizer"] = optimizer.state_dict()
    entrypoint["epoch"] = epoc
    entrypoint["best_acc"] = bess_acc
    
    buffer = io.BytesIO()
    torch.save(entrypoint, buffer)
    buffer.seek(0)
    upload_to_s3(buffer, bucket, s3_key)

# load state dict
def load_model_dict(model: nn.Module, path, optimizer: torch.optim.Optimizer, device: torch.device = torch.device("cpu")):
    s3_key = str(path).replace("\\", "/")
    bucket = settings.AWS_S3_MODELS_BUCKET
    
    print(f"Loading checkpoint from S3: s3://{bucket}/{s3_key}")
    buffer = download_from_s3(bucket, s3_key)
    entrypoint = torch.load(buffer, map_location=device)
    
    model.load_state_dict(entrypoint["model"])
    optimizer.load_state_dict(entrypoint["optimizer"])
    return entrypoint["epoch"], entrypoint["best_acc"]
