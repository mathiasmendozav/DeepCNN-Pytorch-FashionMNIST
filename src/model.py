#####################################
# Convolutional Neural Network Models
#####################################
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
# 6 Million Parameter Model (6.6 Million to be precise)
class ComplexCNNModel(nn.Module):
    def __init__(self, input_channels: int = 1, output_shape: int = 10):
        super(ComplexCNNModel, self).__init__()
        
        # Initial convolution layer
        self.initial_conv = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        
        # Define res_block as a class method
        def res_block(in_channels, out_channels, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False) if in_channels != out_channels or stride != 1 else nn.Identity(),
            )
        
        # creating layer blocks
        self.layer1 = self._make_layer(32, 64, 2, stride=1, block=res_block)
        self.layer2 = self._make_layer(64, 128, 3, stride=2, block=res_block)
        self.layer3 = self._make_layer(128, 256, 3, stride=2, block=res_block)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, block=res_block)

        # adaptive pooling and final classifier and fully connected layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, output_shape)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride=1, block=None):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_conv(x)
        x = F.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x