# Libraries and functions to train the AdaIN network
import argparse
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from PIL import Image, ImageFile
from torchvision import transforms
import AdaIN_net as net

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Function to transform images for training
def train_transform():
    transform_list = [
        # Setting all images to the same size
        transforms.Resize(size=(512, 512)),
        # Randomly cropping the images to 256x256 pixels to capture the most important features
        transforms.RandomCrop(256),
        # PIL to Tensor
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

# Function to normalize losses
def normalize(loss_list):
    # Convert to numpy array for easier math handling
    loss_array = np.array(loss_list)
    # Z-score normalization
    normalized = (loss_array - loss_array.mean()) / loss_array.std()
    # Convert back to python list
    return normalized.tolist()

# Dataset class for loading images from a folder
class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        # Get all the images in the folder
        self.root = root
        # Transform the images
        self.transform = transform
        # Get all the paths to the images
        self.paths = list(Path(self.root).glob('*'))

    # Function to get the image at a given index
    def __getitem__(self, index):
        path = self.paths[index]
        try:
            img = Image.open(str(path)).convert('RGB')
            img = self.transform(img)
            return img
        except Exception as e:
            print(f"Error loading image {path}: {str(e)}")
            # Return a placeholder image or handle the error in a way that fits your needs
            return torch.zeros(3, 256, 256)  # Placeholder image

    # Function to get the length of the dataset
    def __len__(self):
        return len(self.paths)
    # Function to get the name of the dataset
    def name(self):
        return 'FlatFolderDataset'

# Function to adjust the learning rate
def adjust_learning_rate(optimizer, iteration_count, initial_lr = 1e-4, lr_decay = 5e-5):
    # Compute the new learning rate using an inverse decay formula.
    lr = initial_lr / (1.0 + lr_decay * iteration_count)
    # Update the learning rate for all parameter groups in the optimizer.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Function to plot the losses
def loss_plot(c_losses, s_losses, t_losses, save_path):
    # Figure size
    plt.figure(figsize=(10, 6))
    plt.plot(t_losses, label='Content + Style', color='blue')
    plt.plot(c_losses, label='Content', color='orange')
    plt.plot(s_losses, label='Style', color='green')
    # Plotting the legend
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(save_path)
    


# Main function
if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser(description="AdaIN Style + Content Transfer")
    # Image directories
    parser.add_argument('-content_dir', type=str, required=True, help='Directory path to a batch of content images')
    parser.add_argument('-style_dir', type=str, required=True, help='Directory path to a batch of style images')
    # Hyperparameters
    parser.add_argument('-gamma', type=float, default=1.0, help='Gamma parameter')
    parser.add_argument('-e', type=int, default=25, help='Number of epochs')
    parser.add_argument('-b', type=int, default=5, help='Batch size')
    parser.add_argument('-l', type=str, default='encoder.pth', help='Load encoder')
    parser.add_argument('-s', type=str, default='decoder.pth', help='Save decoder')
    parser.add_argument('-p', type=str, default='decoder.png', help='Path to save the loss plot')
    parser.add_argument('-cuda', type=str, choices=['Y', 'N'], help='Use CUDA if available', default='Y')
    args = parser.parse_args()
    
    # Setting the device
    device = torch.device("cuda" if args.cuda == 'Y' and torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Loading network and datasets
    encoder = net.encoder_decoder.encoder
    encoder.load_state_dict(torch.load(args.l, map_location=device))
    network = net.AdaIN_net(encoder)
    network.to(device).train()
    
    # Setting up the dataloaders
    content_tf, style_tf = train_transform(), train_transform()
    COCO_dataset = FlatFolderDataset(args.content_dir, content_tf)
    wikiart_dataset = FlatFolderDataset(args.style_dir, style_tf)
    
    # Get the actual size of the datasets based on the directory
    num_images_coco = len(COCO_dataset)
    num_images_wikiart = len(wikiart_dataset)
    
    # Getting the subset of the datasets
    COCO_dataset = data.Subset(COCO_dataset, list(range(num_images_coco)))
    wikiart_dataset = data.Subset(wikiart_dataset, list(range(num_images_wikiart)))

    content_loader = data.DataLoader(COCO_dataset, batch_size=args.b, shuffle=True, num_workers= 2)
    style_loader = data.DataLoader(wikiart_dataset, batch_size=args.b, shuffle=True, num_workers= 2)

    # Initializing the losses
    content_losses, style_losses, total_losses = [], [], []
    avg_content_losses, avg_style_losses, avg_total_losses = [], [], []

    # Setting up the optimizer
    optimizer = torch.optim.Adam(network.decoder.parameters(), lr= 1e-4)

    # Training loop
    for epoch in range(args.e):
        adjust_learning_rate(optimizer, iteration_count=epoch)
        # Printing the epoch
        print(f"Epoch {epoch + 1}/{args.e}")
        # Initializing the epoch losses
        epoch_content_loss, epoch_style_loss, epoch_total_loss = 0, 0, 0
        epoch_start = time.time()

        for content_images, style_images in zip(content_loader, style_loader):
            # Setting the images to the device
            content_images = content_images.to(device)
            style_images = style_images.to(device)
            
            # Zeroing the gradients
            optimizer.zero_grad()

            # Forward pass
            loss_c, loss_s, _ = network(content_images, style_images)
            
            # Total loss with the gamma parameter
            total_loss = loss_c + args.gamma * loss_s
            total_loss.backward()
            
            # Updating the weights
            optimizer.step()
            
            # Appending the losses
            content_losses.append(loss_c.item())
            style_losses.append(loss_s.item())
            total_losses.append(total_loss.item())

            # Updating the epoch losses
            epoch_content_loss += loss_c.item()
            epoch_style_loss += loss_s.item()
            epoch_total_loss += total_loss.item()

        num_loader = len(content_loader)

        # Averaging the losses
        avg_content_loss = sum(content_losses[-num_loader:]) /  num_loader
        avg_style_loss = sum(style_losses[-len(style_loader):]) /  num_loader
        avg_total_loss = sum(total_losses[- num_loader:]) /  num_loader
        avg_content_losses.append(avg_content_loss)
        avg_style_losses.append(avg_style_loss)
        avg_total_losses.append(avg_total_loss)
        
        epoch_end = time.time()

        # Printing the Parameters
        print(f"Epoch {epoch + 1}/{args.e} - "
          f"Content Loss: {avg_content_loss:.2f}, Style Loss: {avg_style_loss:.2f}, "
          f"Total Loss: {avg_total_loss:.2f} - "
          f"Time: {epoch_end - epoch_start:.2f} seconds")

        # Saving the model
        torch.save(net.encoder_decoder.decoder.state_dict(), args.s)

    # Normalizing the losses
    normalized_content_losses = normalize(avg_content_losses)
    normalized_style_losses = normalize(avg_style_losses)
    normalized_total_losses = normalize(avg_total_losses)
    
    # Plotting the losses
    loss_plot(avg_content_losses, avg_style_losses, avg_total_losses, args.p)