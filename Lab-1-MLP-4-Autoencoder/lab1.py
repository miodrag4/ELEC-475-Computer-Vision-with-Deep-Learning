import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from Model import AutoencoderMLP4Layer
from Train import train_model
    
#Step 4--------------------------------------------------------------------------------------------
# Load the pretrained model
model = AutoencoderMLP4Layer(N_bottleneck=8)
# Load the pretrained weights
model.load_state_dict(torch.load('autoencoder_model.pth'))
# Set the model in evaluation mode (disable gradient calculations)
model.eval()

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
dataset = MNIST('./data/mnist', train=False, download=True, transform=transform)
# Choose an index for the image you want to test
image_index = 101 
# Get the input image
input_image, _ = dataset[image_index]
# Flatten and convert to float32
input_image = input_image.view(1, -1).to(torch.float32)
# Normalize to [0, 1]
input_image /= input_image.max() # need clarification
# Disable gradient calculations during inference, same as model.eval() just good practice.
with torch.no_grad():
    # Pass the input image through the model to get the output
    output_image = model(input_image)
# Decode the flattened image to the original image shape
# Reshape the output to match the input image shape
output_image = output_image.view(28, 28).cpu().numpy()
# Convert tensors to NumPy arrays for visualization
input_image = input_image.view(28, 28).cpu().numpy()
# Display the input and output images side-by-side
f = plt.figure()
f.add_subplot(1, 2, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Input Image')
f.add_subplot(1, 2, 2)
plt.imshow(output_image, cmap='gray')
plt.title('Reconstructed Image')
plt.show()

#Step 5--------------------------------------------------------------------------------------------
# Choose an index for the image you want to test
image_index = 101  # Change this to the desired index
# Get the input image
input_image, _ = dataset[image_index]
# Flatten and convert to float32
input_image = input_image.view(1, -1).to(torch.float32)
# Normalize to [0, 1]
input_image /= input_image.max() # need clarification
# Add uniform noise to the clean input image
noise_level = 1
# This part generates random noise with the same shape as the input_image. 
# It uses torch.rand_like to create a tensor of random values in the range [0, 1] that matches the shape of input_image.
noisy_image = input_image + noise_level * torch.rand_like(input_image)
# Clip the noisy image to ensure pixel values are in [0, 1], since the noise may have pushed some values outside this range
noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
# Disable gradient calculations during inference, same as model.eval() just good practice.
with torch.no_grad():
    # Pass the noisy input image through the model to get the denoised output
    denoised_output_image = model(noisy_image)
# Reshape the output and input images to match the original image shape
denoised_output_image = denoised_output_image.view(28, 28).cpu().numpy()
noisy_image = noisy_image.view(28, 28).cpu().numpy()
input_image = input_image.view(28, 28).cpu().numpy()
# Display the original input, noisy input, and denoised output side-by-side
f = plt.figure(figsize=(18, 6))
f.add_subplot(1, 3, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Original Input Image')
f.add_subplot(1, 3, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Input Image')
f.add_subplot(1, 3, 3)
plt.imshow(denoised_output_image, cmap='gray')
plt.title('Denoised Output Image')
plt.show()

#Step 6--------------------------------------------------------------------------------------------

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