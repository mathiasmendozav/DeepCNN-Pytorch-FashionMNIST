#################
# Testing Model
#################
import torch
import torch.nn as nn
from model import CNNModel


# loading saved state dict model
model = CNNModel(input_channels=1, output_shape=10)
model.load_state_dict(torch.load('/saved_models/1.8M-CNNModel.pth'))

if __name__ == "__main__":
    