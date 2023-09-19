import torch.nn as nn
import torch as torch
import torch.nn.functional as F

# Model class definition (Step 1)
class AutoencoderMLP4Layer(nn.Module):
    def __init__(self, N_input=784, N_bottleneck=8, N_output=784):
        super(AutoencoderMLP4Layer, self).__init__()
        N2 = 392
        self.fc1 = nn.Linear(N_input, N2)
        self.fc2 = nn.Linear(N2, N_bottleneck)
        self.fc3 = nn.Linear(N_bottleneck, N2)
        self.fc4 = nn.Linear(N2, N_output)
    
    # x = self.fc1(x): This line passes the input tensor x through the first fully connected (linear) layer self.fc1. 
    # The linear layer transforms the input by performing a linear combination of its weights and biases. 
    # It's essentially applying an affine transformation to the input data.
    
    # Relu introduces non-linearity by setting negative values to zero and keeping positive values unchanged.
    
    # x = self.fc2(x): The next step is passing the tensor through the second fully connected layer self.fc2. 
    # This layer further transforms the data.
    
    # x = torch.sigmoid(x): Finally, the output of the fourth linear layer is passed through the sigmoid activation function. 
    # Sigmoid squashes the values between 0 and 1, making the output suitable for representing pixel values in an image.
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x
    
    # Step 6
    def encode(self, x): 
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def decode(self, x): 
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x