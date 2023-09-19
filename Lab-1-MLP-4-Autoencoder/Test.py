import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from Model import AutoencoderMLP4Layer  # Import your model class from 'model.py'

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