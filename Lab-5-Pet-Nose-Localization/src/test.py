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
from PIL import ImageDraw

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
    
transform = transforms.Compose([
    transforms.Lambda(lambda image: image.convert('RGB')),  # Convert grayscale images to RGB
    transforms.Resize((224, 224)),  # Resize to the input size of the model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, rmse, mae

# Load your trained model
model_path = "C:/Users/miles/OneDrive - Queen's University/Eng Year 4 - 2024/CMPE 475/Lab-5-Pet-Nose-Localization/src/new_nose_localization_model.pth"  # Update this path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNNWithRegressionHead()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()  # Set the model to evaluation mode

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare your test data loader
test_data_path = "C:/Users/miles/OneDrive - Queen's University/Eng Year 4 - 2024/CMPE 475/Lab-5-Pet-Nose-Localization/src/154"
test_dataset = NoseDataset(image_dir=test_data_path, 
                           labels_file=os.path.join(test_data_path, 'labels.txt'), 
                           transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Function to denormalize the coordinates
def denormalize_coordinates(coords, img_size):
    x, y = coords
    img_width, img_height = img_size
    return int(x * img_width), int(y * img_height)

# Run the model on the test data
all_labels = []
all_preds = []

# Run the model on the test data and visualize the results
for i, (inputs, labels) in enumerate(test_loader):
    inputs = inputs.to(device)
    labels = labels.to(device)
    predicted_coords = model(inputs).cpu().detach().numpy()[0]  # Remove batch dimension

    # Get the original image size
    original_img_path = os.path.join(test_data_path, test_dataset.labels[i][0])
    original_img = Image.open(original_img_path)
    original_size = original_img.size  # (width, height)

    # Denormalize the predicted coordinates
    predicted_coords = denormalize_coordinates(predicted_coords, original_size)
    
    # Denormalize the true label coordinates
    true_coords = denormalize_coordinates(labels.cpu().numpy()[0], original_size)
    
    # Append results for evaluation
    all_labels.append(labels.cpu().numpy())
    all_preds.append(predicted_coords)

    # Draw the predicted and true points on the original image
    draw = ImageDraw.Draw(original_img)
    draw.point(predicted_coords, 'red')
    draw.point(true_coords, 'green')

    # Display the original image with the true and predicted points
    fig, ax = plt.subplots()
    ax.imshow(original_img)
    ax.plot(*predicted_coords, 'ro', markersize=10, label='Predicted')  # Predicted point
    ax.plot(*true_coords, 'go', markersize=10, label='True Label')  # True point
    ax.set_xlim(0, original_size[0])
    ax.set_ylim(original_size[1], 0)  # Inverted y-axis for correct orientation
    ax.axis('off')  # Hide the axis
    plt.legend()
    plt.show()

    # Optionally, save the image with annotations
    draw = ImageDraw.Draw(original_img)
    draw.point(true_coords, fill='green')
    draw.point(predicted_coords, fill='red')
    
# After processing all test data
all_labels = np.concatenate([label.cpu().numpy() for label in all_labels], axis=0)
all_preds = np.concatenate(all_preds, axis=0)

# Ensure labels are denormalized if necessary before calculating metrics
all_labels_denorm = [denormalize_coordinates(label, original_size) for label in all_labels]
all_labels_denorm = np.array(all_labels_denorm)

mse, rmse, mae = calculate_metrics(all_labels_denorm, all_preds)
print(f'Test MSE: {mse:.4f}')
print(f'Test RMSE: {rmse:.4f}')
print(f'Test MAE: {mae:.4f}')