import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image, ImageFile


class custom_dataset(Dataset):
    def __init__(self, dir, transform=None):
        super().__init__()
        Image.MAX_IMAGE_PIXELS = None
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.transform = transform
        self.image_files = [os.path.join(dir, file_name) for file_name in os.listdir(dir) if os.path.isfile(os.path.join(dir, file_name)) and file_name.endswith(".png")]
        self.labels = open(os.path.join(dir, 'labels.txt')).read().splitlines()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = Image.open(self.image_files[index]).convert('RGB')
        label = int(self.labels[index].split(' ')[1])
        # print(label)
        image_sample = self.transform(image)
        return image_sample, label

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import argparse
# Parsing arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-e', '--epochs', type=int, default=50, help='Training epochs')
arg_parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
arg_parser.add_argument('-s', '--save_path', type=str, default="./output/model.pth", help='Path to save the model')
arg_parser.add_argument('-p', '--plot_path', type=str, default="./output/loss_plot.png", help='Path for loss plot')
arg_parser.add_argument('--use_cuda', type=str, default='n', help='Use CUDA [y/N]')
args = arg_parser.parse_args()

# Data transformations with normalization (without augmentation)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Datasets and DataLoaders
train_dataset = custom_dataset(dir='./data/Kitti8_ROIs/train', transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

# Initialize the model with pretrained weights
model = models.resnet18()
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 2)
)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Device configuration
use_cuda = args.use_cuda.lower() == 'y'
device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
num_epochs = args.epochs
train_loss_list = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Average training loss
    train_loss /= len(train_loader)
    train_loss_list.append(train_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')

    # Adjust learning rate
    scheduler.step()

# Plot the training loss
plt.figure()
plt.plot(train_loss_list, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss over Epochs')
plt.savefig(args.plot_path)
plt.show()

# Save the model
torch.save(model.state_dict(), args.save_path)

# Evaluate the model on the training dataset
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f'Accuracy on Training Data: {accuracy:.4f}')

# Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
print('Confusion Matrix on Training Data:\n', conf_matrix)
