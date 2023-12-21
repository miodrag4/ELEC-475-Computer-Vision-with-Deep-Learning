import torch
import torch.nn as nn
from torchvision import models

class RegressionHead(nn.Module):
    def __init__(self, num_features, dropout_probability=0.5):
        super(RegressionHead, self).__init__()
        self.dropout = nn.Dropout(dropout_probability)
        self.fc = nn.Linear(num_features, 2)  # 2 for x and y coordinates
        self.batch_norm = nn.BatchNorm1d(2)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        x = self.batch_norm(x)
        return x

class CustomCNNWithRegressionHead(nn.Module):
    def __init__(self, fine_tune=False):
        super(CustomCNNWithRegressionHead, self).__init__()
        # Load a pre-trained ResNet model
        self.base_model = models.resnet18(pretrained=True)

        # If fine-tuning, set requires_grad to True
        if fine_tune:
            for param in self.base_model.parameters():
                param.requires_grad = True
        else:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Remove the original classification head
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()

        # Add the custom regression head
        self.regression_head = RegressionHead(num_features)

    def forward(self, x):
        # Pass input through the base model
        features = self.base_model(x)
        # Pass the features through the regression head
        output = self.regression_head(features)
        return output

# Create an instance of the model
model = CustomCNNWithRegressionHead(fine_tune=True)