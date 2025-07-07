import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(362112, 64)
        self.fc2 = nn.Linear(64, 17)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def train(model, train_loader, loss_fct, optimizer, log_interval, device):
    # Set the model to training mode
    model.train()

    running_loss = 0.0
    for batch_id, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = loss_fct(output, target)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        # print(loss.item())
        
        # Print Statistics
        running_loss += loss.item()
        if batch_id % log_interval == log_interval-1:    
            print(f'[{batch_id + 1:5d}/{len(train_loader):>5d}] Loss: {running_loss / 2000:.5f}')
            running_loss = 0.0


def test(test_loader, model, loss_fct, device):
    # Set the model to evaluation mode
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fct(output, target).item()  # sum up batch loss
            

    test_loss /= len(test_loader)

    print(f'Test set: Average loss: {test_loss:.4f}')




