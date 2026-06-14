import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import classification_report

class BaseModel(nn.Module):
    """
    Abstract Base Model Wrapper defining the contract for all neural networks
    in the Knee Osteoarthritis classification system.
    """
    def __init__(self):
        super(BaseModel, self).__init__()
        
    def forward(self, x):
        pass
        
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


