#################
# Training Model
#################
import os
import torch
import torch.nn as nn
from model import CNNModel6M
from tqdm import tqdm
from dataset import get_data_loaders
from torchmetrics import Accuracy

# Ensure the directory for saving model weights exists
SAVE_DIR = os.path.join(os.getcwd(), "saved_models")
os.makedirs(SAVE_DIR, exist_ok=True)  # Create 'saved_models' folder if it doesn't exist

# train function
def training_loop(model: nn.Module, dataloader: list, loss_fn: nn.Module, optimizer: torch.optim, device: torch.device):
    train_loss = 0
    # Adding tqdm to show progress for each batch
    loop = tqdm(dataloader, desc="Training", leave=False)
    for batch, (X_train, y_train) in enumerate(loop):
        X_train, y_train = X_train.to(device), y_train.to(device)
        
        # forward pass
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        train_loss += loss.item()  # convert loss tensor to a scalar value
        
        # gradient descent
        optimizer.zero_grad()
        # back propagation
        loss.backward()
        # optimizer step
        optimizer.step()

        # Update progress description with current batch and loss
        loop.set_postfix(loss=loss.item())
            
    train_loss /= len(dataloader)  # average loss over all batches
    return train_loss
            

# eval function
def testing_loop(model: nn.Module, dataloader: list, loss_fn: nn.Module, acc_fn, device: torch.device):
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        # Adding tqdm to show progress for evaluation
        loop = tqdm(dataloader, desc="Testing", leave=False)
        for X_test, y_test in loop:
            X_test, y_test = X_test.to(device), y_test.to(device)
            
            # forward pass
            y_pred = model(X_test)
            test_loss += loss_fn(y_pred, y_test).item()  # convert loss tensor to a scalar value
            test_acc += acc_fn(y_pred.argmax(dim=1), y_test)

            # Update progress description with current test loss
            loop.set_postfix(loss=test_loss)

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        
    return test_loss, test_acc


if __name__ == '__main__':
    # Get data loaders
    train_dataloader, test_dataloader = get_data_loaders()
    
    # get cuda device (GPU) or cpu if not available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Working on: {torch.cuda.get_device_name(torch.cuda.current_device())}')
    
    # create new instance of model or load saved model weights
    model = CNNModel6M(input_channels=1, output_shape=10)
    model.to(device)
    
    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # define accuracy function (using torchmetrics in this case)
    acc_fn = Accuracy(task='multiclass', num_classes=10).to(device)
    
    # define training and eval flow
    epochs = 6
    for epoch in tqdm(range(epochs), desc="Epoch Progress"):
        # training
        train_loss = training_loop(model, train_dataloader, loss_fn, optimizer, device)
        # testing
        test_loss, eval_acc = testing_loop(model, test_dataloader, loss_fn, acc_fn, device)

        print(f'Epoch: {epoch+1}/{epochs} | Train Loss: {train_loss:.2f} | Test Loss: {test_loss:.2f} | Eval Acc: {eval_acc*100:.2f}%')

        # Save the model weights at the end of each epoch
        model_save_path = os.path.join(SAVE_DIR, f"CNNModel-6M.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")