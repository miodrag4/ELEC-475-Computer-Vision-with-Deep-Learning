import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import argparse
import time

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, labels_filename, transform=None):
        # Ensure the base directory is an absolute path
        self.image_dir = os.path.abspath(image_dir)
        # The labels file is directly in the image_dir
        labels_file = os.path.join(self.image_dir, labels_filename)
        self.transform = transform
        self.img_labels = []
        
        with open(labels_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                image_name, _, label_str = line.strip().split()
                full_image_path = os.path.join(self.image_dir, image_name)
                label_int = self.label_to_int(label_str) 
                self.img_labels.append((full_image_path, label_int))
                
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image_path, label = self.img_labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    @staticmethod
    def label_to_int(label_str):
        return {'NoCar': 0, 'Car': 1}.get(label_str, -1)

# Paths
TRAIN_DATA_PATH = "C:\\Users\\miles\\OneDrive - Queen's University\\Eng Year 4 - 2024\\CMPE 475\\Lab-4-YODA-You-Only-Detect-Anchors\\data\\Kitti8_ROIs\\train"
TEST_DATA_PATH = "C:\\Users\\miles\\OneDrive - Queen's University\\Eng Year 4 - 2024\\CMPE 475\\Lab-4-YODA-You-Only-Detect-Anchors\\data\\Kitti8_ROIs\\test"
TRAIN_LABELS_FILE = os.path.join(TRAIN_DATA_PATH, 'labels.txt')
TEST_LABELS_FILE = os.path.join(TEST_DATA_PATH, 'labels.txt')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train and evaluate a ResNet18 model on a custom dataset.')
    parser.add_argument('--train_data_path', type=str, required=True, help='Path to the training data')
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to the test data')
    parser.add_argument('--num_epochs', type=int, default=40, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for optimizer')
    parser.add_argument('--model_save_path', type=str, default='model.pth', help='Path to save the trained model')
    return parser.parse_args()

args = parse_arguments()

# Use parsed arguments
num_epochs = args.num_epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
train_data_path = args.train_data_path
test_data_path = args.test_data_path
model_save_path = args.model_save_path

# Transforms (assuming they remain constant)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Datasets and DataLoaders (using parsed paths)
train_dataset = CustomDataset(train_data_path, os.path.join(train_data_path, 'labels.txt'), transform=transform)
test_dataset = CustomDataset(test_data_path, os.path.join(test_data_path, 'labels.txt'), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet18(pretrained=True)
model.fc = nn.Sequential(nn.BatchNorm1d(model.fc.in_features), nn.Dropout(0.5), nn.Linear(model.fc.in_features, 2))
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training and Validation
train_losses, val_losses = [], []
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    start_time = time.time()  # Start time for the epoch
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        # Print progress update
        print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Batch Loss: {loss.item():.4f}', end='\r')

    end_time = time.time()  # End time for the epoch
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(test_loader)
    val_losses.append(avg_val_loss)

    scheduler.step(avg_val_loss)
    # Print epoch summary
    print(f'Epoch {epoch+1}/{num_epochs} completed in {end_time - start_time:.2f} seconds, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Save Model
torch.save(model.state_dict(), '2nd-1.pth')

model.eval()  # Evaluation mode
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

confusion = confusion_matrix(all_labels, all_predictions)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')

accuracy = np.sum(np.array(all_labels) == np.array(all_predictions)) / len(all_labels)
print(f'Accuracy: {accuracy:.4f}')
print(f'Confusion Matrix:\n{confusion}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')