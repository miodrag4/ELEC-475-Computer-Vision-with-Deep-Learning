import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import datetime as datetime

# Define the training function
def train_model(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device):
    print("Training...")
    model.train()  # Keeps track of gradient for backpropagation
    losses_train = [] # List for saving losses per epoch

    for epoch in range(1, n_epochs + 1):
        print('Epoch', epoch)
        loss_train = 0.0
        
        for batch in train_loader:
            imgs, _ = batch
            imgs = imgs.view(imgs.size(0), -1).to(device)  # Flatten the images
            outputs = model(imgs)                          # forward propagation through the model
            loss = loss_fn(outputs, imgs)                  # calculate the loss
            optimizer.zero_grad()                          # clear the gradients
            loss.backward()                                # calculate the loss gradients
            optimizer.step()                               # iterate the optimization, based on loss gradients
            loss_train += loss.item()                      # update the training loss

        scheduler.step(loss_train)  # Adjust the learning rate
        losses_train.append(loss_train / len(train_loader))
        print(f'Epoch {epoch}/{n_epochs}, Training Loss: {loss_train / len(train_loader):.4f}')

    return losses_train