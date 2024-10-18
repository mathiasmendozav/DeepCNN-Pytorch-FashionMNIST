#################
# Testing Model
#################
import torch
import torch.nn as nn
from model import CNNModel
from dataset import get_data_loaders
from utils import get_evaluation_data
import random

# loading saved state dict model
model = CNNModel(input_channels=1, output_shape=10)

if __name__ == "__main__":
    # getting dataloaders
    test_data = get_evaluation_data()
    
    # getting random samples from test_data for evaluation
    test_samples = []
    test_labels = []
    for sample, label in random.sample(list(test_data), k=9):
        test_samples.append(sample)
        test_labels.append(label)
    