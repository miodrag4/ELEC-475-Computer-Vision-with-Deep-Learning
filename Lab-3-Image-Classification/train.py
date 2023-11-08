import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vgg_model import encoder_decoder
from model import VGGBackend, Frontend, VanillaModelv1, ResidualBlock, ResNetBackend, VanillaModelv2, Frontendv2
import matplotlib.pyplot as plt

def main():

    # CIFAR-100: Mean and Std for normalization
    mean_cifar100 = [0.5071, 0.4867, 0.4408]
    std_cifar100 = [0.2675, 0.2565, 0.2761]

    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean_cifar100, std_cifar100),
    ])

    # Just normalization for validation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_cifar100, std_cifar100),
    ])

    # Load CIFAR-100 datasets
    trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = datasets.CIFAR100(root='./data', train=False,
                                download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100,
                            shuffle=False, num_workers=2)
    # V1
    # # Initialize backend and frontend
    # vgg_backend = VGGBackend()  # Assumes the encoder is part of this class as shown previously
    # frontend = Frontend(num_classes=100)
    # model = VanillaModelv1(backend=vgg_backend, frontend=frontend)
    
    # V2
    # Initialize backend and frontend
    resnet_backend = ResNetBackend()  # Assumes the encoder is part of this class as shown previously
    frontend = Frontend(num_classes=100)
    model = VanillaModelv1(backend=resnet_backend, frontend=frontend)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training
    model.train()  # Set the model to training mode
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the parameter gradients

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")

    print('Finished Training')

    # Testing
    model.eval()  # Set the model to evaluation mode
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            # Top 1 accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_top1 += (predicted == labels).sum().item()

            # Top 5 accuracy
            _, top5 = outputs.topk(5, 1, True, True)
            correct_top5 += top5.eq(labels.view(-1, 1).expand_as(top5)).sum().item()

    print('Accuracy of the network on the 10000 test images:')
    print(f"Top 1 accuracy: {100 * correct_top1 / total}%")
    print(f"Top 5 accuracy: {100 * correct_top5 / total}%")    
    
if __name__ == '__main__': 
    main()