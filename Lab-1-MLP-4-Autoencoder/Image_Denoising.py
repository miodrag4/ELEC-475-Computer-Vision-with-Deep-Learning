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
model.eval()  # Set the model in evaluation mode (disable gradient calculations)

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
dataset = MNIST('./data/mnist', train=False, download=True, transform=transform)

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