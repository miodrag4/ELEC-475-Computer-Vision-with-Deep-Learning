import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from model import CustomCNNWithRegressionHead
import os
from PIL import Image
import ast
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

# Define transformations with normalization parameters if available
transform = transforms.Compose([
    transforms.Lambda(lambda image: image.convert('RGB')),  # Convert grayscale images to RGB
    transforms.Resize((224, 224)),  # Consider using a different size if needed
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
])

train_dataset = NoseDataset(image_dir=args.image_dir, 
                            labels_file=args.train_labels, 
                            transform=transform)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)


val_dataset = NoseDataset(image_dir=args.image_dir, 
                          labels_file=args.val_labels, 
                          transform=transform)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

class EuclideanDistanceLoss(nn.Module):
    def forward(self, pred, target):
        return torch.sqrt(torch.sum((pred - target) ** 2, dim=1)).mean()

# Model, Loss and Optimizer
model = CustomCNNWithRegressionHead(fine_tune=True)
criterion = EuclideanDistanceLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, rmse, mae

# Training and validation loop
num_epochs = args.epochs
train_losses, val_losses = [], []
# Define the scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Initialize early stopping parameters
early_stopping_patience = 10
early_stopping_counter = 0
best_val_loss = float('inf')

for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    # Validation
    model.eval()
    running_val_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())
            val_loss = criterion(outputs, labels)
            running_val_loss += val_loss.item()
    val_losses.append(running_val_loss / len(val_loader))

    # Calculate metrics
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    mse, rmse, mae = calculate_metrics(all_labels, all_preds)

    # Scheduler step and early stopping check
    scheduler.step(val_losses[-1])
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        early_stopping_counter = 0
        torch.save(model.state_dict(), 'nose_localization_model.pth')
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    end_time = time.time()
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_losses[-1]:.4f}, 'f'Validation Loss: {val_losses[-1]:.4f},MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, 'f'Time: {end_time - start_time:.2f} seconds')

# Determine the number of epochs that actually occurred
actual_epochs = len(train_losses)

# Plotting the loss over epochs
plt.figure(figsize=(10,5))  # Optional: You can specify the figure size
plt.plot(range(1, actual_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, actual_epochs + 1), val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the final model
torch.save(model.state_dict(), 'nose_localization_model.pth')