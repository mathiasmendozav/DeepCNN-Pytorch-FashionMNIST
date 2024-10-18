#################
# Testing Model
#################
import torch
import torch.nn as nn
from model import CNNModel
from utils import get_evaluation_data
import random
from tqdm import tqdm
from torchmetrics import Accuracy
import os

# Get the current working directory
current_directory = os.getcwd()
# Get current device
device = "cuda" if torch.cuda.is_available() else "cpu"
# Construct the path to the saved model using the current directory
model_path = os.path.join(current_directory, 'saved_models', 'CNNModel-1.8M.pth')
# Initialize model
model = CNNModel(input_channels=1, output_shape=10)
model.to(device)

# Check if the model file exists
if os.path.exists(model_path):
    print(f"Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path))
else:
    raise FileNotFoundError(f"No such file: {model_path}")

if __name__ == "__main__":
    # Get test dataloaders
    test_loader = get_evaluation_data()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = nn.CrossEntropyLoss()

    # Setup accuracy function
    acc_fn = Accuracy(task='multiclass', num_classes=10).to(device)

    # Sample random batches from test_loader for evaluation
    test_samples = random.sample(list(test_loader), k=9)

    # Evaluate model on test samples
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        loop = tqdm(test_samples, desc="Testing", leave=False)
        for X_test, y_test in loop:
            # Moving data to device
            X_test, y_test = X_test.to(device), y_test.to(device)
            # Forward pass
            y_pred = model(X_test)
            # Calculate loss and accuracy
            test_loss += loss_fn(y_pred, y_test).item()  # Convert loss tensor to scalar value
            test_acc += acc_fn(y_pred.argmax(dim=1), y_test)

            # Update progress description with current test loss
            loop.set_postfix(loss=test_loss)

        # Calculate average loss and accuracy
        test_loss /= len(test_samples)
        test_acc /= len(test_samples)
        
        print(f'Test Loss: {test_loss:.2f} | Eval Acc: {test_acc*100:.2f}%')
