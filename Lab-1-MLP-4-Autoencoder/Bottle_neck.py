import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from Model import AutoencoderMLP4Layer

# Load the pretrained model
model = AutoencoderMLP4Layer(N_bottleneck=8)
# Load the pretrained weights
model.load_state_dict(torch.load('autoencoder_model.pth'))
# Set the model in evaluation mode (disable gradient calculations)
model.eval()

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
dataset = MNIST('./data/mnist', train=False, download=True, transform=transform)

# Choose two input images for interpolation
image_index_1 = 78  # This will be the first starting image
image_index_2 = 880  # This is what the first image will be linearly interpolated to

# Get the images and flatten them
image_1, _ = dataset[image_index_1]
image_2, _ = dataset[image_index_2]
# Flatten and convert to float32
image_1 = image_1.view(1, -1).to(torch.float32)
image_2 = image_2.view(1, -1).to(torch.float32)

# Encode the two input images to get their bottleneck representations
bottleneck_1 = model.encode(image_1)
bottleneck_2 = model.encode(image_2)

# Define the number of interpolation steps
n_steps = 9

# This code is performing linear interpolation between bottleneck_1 and bottleneck_2 by 
# varying the alpha value, and at each step, it calculates the interpolated bottleneck representation. 
# This allows you to generate a sequence of bottleneck representations that smoothly transition from one to the other.
interpolated_bottlenecks = []
for i in range(n_steps):
    alpha = i / (n_steps - 1)
    interpolated_bottleneck = alpha * bottleneck_1 + (1 - alpha) * bottleneck_2
    interpolated_bottlenecks.append(interpolated_bottleneck)

# Decode the list of 10 interpolated bottleneck representations to get reconstructed images
interpolated_images = [model.decode(bottleneck).view(28, 28).cpu().detach().numpy() for bottleneck in interpolated_bottlenecks]

# Display the original input images and the interpolated images
f = plt.figure(figsize=(12, 4))
# Plot the first input image, followed by the interpolated images, and finally the second input image
for i, image in enumerate([image_2.view(28, 28).cpu().numpy()] + interpolated_images + [image_1.view(28, 28).cpu().numpy()]):
    f.add_subplot(1, n_steps + 2, i + 1)
    plt.imshow(image, cmap='gray')
    
plt.show()