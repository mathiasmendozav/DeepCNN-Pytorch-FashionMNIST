#####################################
# Convolutional Neural Network Model
#####################################
import torch
import torch.nn as nn

# Model class
class CNNModel(nn.Module):
    def __init(self, input_shape: int, hidden_units: int, output_shape: int):
        super(CNNModel, self).__init__()
        
        # conv blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                                out_channels=hidden_units,
                                kernel_size=3,
                                padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_units),
            nn.Conv2d(in_channels=hidden_units,
                                out_channels=hidden_units,
                                kernel_size=3,
                                padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_units),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                                out_channels=hidden_units,
                                kernel_size=3,
                                padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_units),
            nn.Conv2d(in_channels=hidden_units,
                                out_channels=hidden_units,
                                kernel_size=3,
                                padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_units),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )