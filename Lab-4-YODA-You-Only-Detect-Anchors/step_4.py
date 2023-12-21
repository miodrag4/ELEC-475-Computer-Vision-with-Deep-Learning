import argparse
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import cv2
import torch.nn as nn
from KittiAnchors import Anchors

# Define a function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the YODA model on KITTI dataset")
    parser.add_argument("--model_weights", type=str, required=True, help="Path to the trained model weights")
    parser.add_argument("--test_image_dir", type=str, required=True, help="Directory containing test images")
    parser.add_argument("--test_label_dir", type=str, required=True, help="Directory containing test labels")
    parser.add_argument("--prob_threshold", type=float, default=0.5, help="Probability threshold for car detection")
    return parser.parse_args()

args = parse_args()

# Set up device, model, and load weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.BatchNorm1d(num_ftrs),
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 2)
)
model.load_state_dict(torch.load(args.model_weights, map_location=device))
model = model.to(device)
model.eval()

# Instantiate the Anchors class
anchors = Anchors()

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess images
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image

# Load ground truth bounding boxes
def load_ground_truth(label_path):
    gt_boxes = []
    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if 'Car' in line:
                top_left = (float(parts[5]), float(parts[4]))
                bottom_right = (float(parts[7]), float(parts[6]))
                gt_boxes.append((top_left, bottom_right))
    return gt_boxes

# Visualize bounding boxes
def visualize(image_np, boxes, title):
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    for box in boxes:
        (y1, x1), (y2, x2) = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
    plt.title(title)
    plt.show()

# Main evaluation loop
iou_scores = []
test_images = [os.path.join(args.test_image_dir, img) for img in os.listdir(args.test_image_dir) if img.endswith('.png')]

for image_path in test_images:
    image = load_image(image_path)
    image_tensor = image.unsqueeze(0).to(device)
    label_path = os.path.join(args.test_label_dir, os.path.basename(image_path).replace('.png', '.txt'))
    gt_boxes = load_ground_truth(label_path)
    image_np = np.array(Image.open(image_path)) 
    anchor_centers = anchors.calc_anchor_centers(image_np.shape, anchors.grid)
    ROIs, boxes = anchors.get_anchor_ROIs(image_np, anchor_centers, anchors.shapes)
    car_boxes = []
    softmax = nn.Softmax(dim=1)
    
    for i, roi in enumerate(ROIs):
        roi_image = Image.fromarray(roi.astype('uint8'), 'RGB')
        roi_tensor = transform(roi_image).unsqueeze(0).to(device)
        outputs = softmax(model(roi_tensor))
        probs = outputs.data[0]
        
        if probs[1] > args.prob_threshold:
            car_boxes.append(boxes[i])
            max_iou = 0
            
            for gt_box in gt_boxes:
                iou = anchors.calc_IoU(boxes[i], gt_box)
                max_iou = max(max_iou, iou)
            iou_scores.append(max_iou)

    visualize(np.array(Image.open(image_path)), car_boxes, "Predicted Cars")

mean_iou = np.mean(iou_scores) if iou_scores else 0
print(f'Mean IoU for detected \'Car\' ROIs: {mean_iou}')