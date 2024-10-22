#####################################
# Convolutional Neural Network Models
#####################################
import torch
import torch.nn as nn

# 2 Million Parameters Model (1.8M to be precise) 
class CNNModel2M(nn.Module):
    """
    Convolutional Neural Network (2 Million parameter model) designed for image classification tasks like FashionMNIST.

    Architecture Overview:
    - Two convolutional blocks, each with:
      - Two layers of convolution followed by ReLU activation and batch normalization
      - Max pooling to reduce spatial dimensions
      - Dropout for regularization
    - A sequence of fully connected layers to map the flattened features to class probabilities

    Attributes:
    - block1 (nn.Sequential): First block of convolutional layers with pooling and dropout
    - block2 (nn.Sequential): Second block, similar to block1 but with increased depth
    - fully_connected_layer (nn.Sequential): Final layers for classification

    Args:
    - input_channels (int): Number of channels in the input image (default is 1 for grayscale)
    - output_shape (int): Number of output classes (default is 10 for FashionMNIST)

    Forward Pass:
    - Applies convolutions, pooling, and dropout, then flattens the output for the fully connected layers
    - Returns logits for each class
    """
    def __init__(self, input_channels: int = 1, output_shape: int = 10):
        super(CNNModel2M, self).__init__()
        
        # First convolutional block
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        # Second convolutional block
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        # Fully connected layers
        self.fully_connected_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128 * 7 * 7, out_features=256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.fully_connected_layer(x)
        return x
    
# 
class CNNModel(nn.Module):
    """
    """
    def __init__(self, input_channels: int = 1, output_shape: int = 10):
        super(CNNModel, self).__init__()
        
        # First convolutional block
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        # Second convolutional block
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        # Third convolutional block
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        # Fourth convolutional block
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        # Fully connected layers
        self.fully_connected_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128 * 7 * 7, out_features=128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.fully_connected_layer(x)
        return x