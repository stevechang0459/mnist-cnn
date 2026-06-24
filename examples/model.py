import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    A lightweight Convolutional Neural Network designed for MNIST digit classification.
    Optimized for Edge AI deployment (compatible with Intel XPU and OpenVINO).
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Block 1: Low-level feature extraction (Edges, simple textures)
        # Input shape:  [Batch, 1, 28, 28]
        # Output shape: [Batch, 16, 28, 28]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        # Downsampling: [Batch, 16, 28, 28] -> [Batch, 16, 14, 14]
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: High-level feature extraction
        # Input shape:  [Batch, 16, 14, 14]
        # Output shape: [Batch, 32, 14, 14]
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        # Downsampling: [Batch, 32, 14, 14] -> [Batch, 32, 7, 7]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Classifier
        # The spatial dimensions are reduced to 7x7 with 32 channels (32 * 7 * 7 = 1568 features)
        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=10) # 10 distinct digit classes (0-9)

    def forward(self, x):
        # Pass through the convolutional blocks
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        # Crucial flattening step:
        # Flattens all dimensions starting from dimension 1 (channel),
        # strictly preserving the 0th dimension (batch size) to prevent shape mismatch errors.
        x = torch.flatten(x, 1)

        # Generate final class logits
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)

        return x
