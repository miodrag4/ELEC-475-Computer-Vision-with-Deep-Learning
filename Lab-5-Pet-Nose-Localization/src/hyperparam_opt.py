import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from model import CustomCNNWithRegressionHead
import optuna
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
import time
from PIL import ImageDraw
import os
from PIL import Image
import ast

parser = argparse.ArgumentParser(description='Train a model for pet nose localization.')
parser.add_argument('--image_dir', type=str, default='src/images', help='Directory where images are stored')
parser.add_argument('--train_labels', type=str, default='src/images/train_noses.txt', help='Path to training labels file')
parser.add_argument('--val_labels', type=str, default='src/images/test_noses.txt', help='Path to validation labels file')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')

args = parser.parse_args()

class NoseDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        self.image_dir = image_dir
        self.labels = self._read_labels_file(labels_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def normalize_coordinates(self, x, y, img_width, img_height):
            return x / img_width, y / img_height

    def __getitem__(self, idx):
        img_name, nose_coords = self.labels[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path)
        img_width, img_height = image.size
        nose_coords = self.normalize_coordinates(*nose_coords, img_width, img_height)
        label = torch.tensor(nose_coords, dtype=torch.float)
        if self.transform:
            image = self.transform(image)
        return image, label

    def _read_labels_file(self, file_path):
        labels = []
        with open(file_path, 'r') as file:
            next(file)  # Skip the header line
            for line in file:
                try:
                    # Strip out new line characters and extra spaces
                    line = line.strip()
                    
                    # Extract the image file name and nose coordinates
                    # Splitting only on the first comma allows for filenames that may contain commas
                    parts = line.split(',', 1)
                    if len(parts) != 2:
                        continue  # Skip lines that do not have the correct format

                    image_name, nose_coords_str = parts
                    # Clean up and parse the coordinates
                    nose_coords_str = nose_coords_str.strip('\"')
                    nose_coords = ast.literal_eval(nose_coords_str)
                    labels.append((image_name, nose_coords))
                except Exception as e:
                    print(f"Error parsing line: {line}. Error: {e}")
                    continue
        return labels

class EuclideanDistanceLoss(nn.Module):
    def forward(self, pred, target):
        return torch.sqrt(torch.sum((pred - target) ** 2, dim=1)).mean()
    
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations with normalization parameters if available
transform = transforms.Compose([
    transforms.Lambda(lambda image: image.convert('RGB')),  # Convert grayscale images to RGB
    transforms.Resize((224, 224)),  # Resize to the input size of the model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Use ImageNet stats
])

# Initialize datasets and dataloaders
train_dataset = NoseDataset(image_dir=args.image_dir, labels_file=args.train_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

val_dataset = NoseDataset(image_dir=args.image_dir, labels_file=args.val_labels, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

def objective(trial):
    # Hyperparameters to optimize
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])

    # Initialize model and optimizer
    model = CustomCNNWithRegressionHead(fine_tune=True).to(device)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    criterion = EuclideanDistanceLoss().to(device)

    # Training and validation loop
    for epoch in range(args.epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation loss
        val_loss = compute_validation_loss(val_loader, model, criterion, device)
        trial.report(val_loss, epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss

def compute_validation_loss(val_loader, model, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
    return val_loss / len(val_loader)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print(f"Best trial: {study.best_trial.params}")

best_trial = study.best_trial
for key, value in best_trial.params.items():
    print(f"{key}: {value}")

# Save the best model
model = CustomCNNWithRegressionHead(fine_tune=True).to(device)
optimizer = getattr(optim, best_trial.params['optimizer'])(model.parameters(), lr=best_trial.params['lr'])
# Load the model weights etc.
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
torch.save(model.state_dict(), 'best_nose_localization_model.pth')