import torch.nn as nn
import torch
from vgg_model import encoder_decoder
import torch.nn.functional as F

# Backend from Lab 2 (VGG-16)
class VGGBackend(nn.Module):
    def __init__(self):
        super(VGGBackend, self).__init__()
        self.encoder = encoder_decoder.encoder
        
    def forward(self, x):
        return self.encoder(x)
    
class Frontend(nn.Module):
    def __init__(self, num_classes=100):
        super(Frontend, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(18432, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Vanilla Model v1
class VanillaModelv1(nn.Module):
    def __init__(self, backend, frontend):
        super(VanillaModelv1, self).__init__()
        self.backend = backend
        self.frontend = frontend
        
    def forward(self, x):
        x = self.backend(x)
        x = self.frontend(x)
        return x

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

# ResNet Backend
class ResNetBackend(nn.Module):
    def __init__(self):
        super(ResNetBackend, self).__init__()
        self.encoder = encoder_decoder.encoder
        self.resblock1 = ResidualBlock(512)
        self.resblock2 = ResidualBlock(512)

    def forward(self, x):
        x = self.encoder(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        return x

class Frontendv2(nn.Module):
    def __init__(self, num_classes=100):
        super(Frontend, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 1 * 1, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x

class VanillaModelv2(nn.Module):
    def __init__(self, backend, frontend):
        super(VanillaModelv1, self).__init__()
        self.backend = backend
        self.frontend = frontend

    def forward(self, x):
        x = self.backend(x)
        x = self.frontend(x)
        return x