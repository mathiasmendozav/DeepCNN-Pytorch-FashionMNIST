#################
# Training Model
#################
import torch
import torch.nn as nn
from model import CNNModel
from tqdm.auto import tqdm
from dataset import get_data_loaders
import torchmetrics

# train function
def training_loop(model: nn.Module, dataloader: list, loss_fn: nn.Module, optimizer: torch.optim, device: torch.device):
    for batch, (X_train, y_train) in enumerate(dataloader):
        X_train, y_train = X_train.to(device), y_train.to(device)
        # forward pass
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        # gradient descent
        optimizer.zero_grad()
        # back propagation
        loss.backward()
        # optimizer step
        optimizer.step()
        
        if batch % 400 == 0:
            print(f'Looked at {batch*len(X_train)}/{len(dataloader.dataset)} samples')
            
    return loss
            
# eval function
def testing_loop(model: nn.Module, dataloader: list, loss_fn: nn.Module, acc_fn: torchmetrics, device: torch.device):
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for X_test, y_test in dataloader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            # forward pass
            y_pred = model(X_test)
            test_loss += loss_fn(y_pred, y_test)
            test_acc += acc_fn(y_pred.argmax(dim=1), y_test)

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        
    return test_loss, test_acc


if __name__ == '__main__':
    # Get data loaders
    train_dataloader, test_dataloader = get_data_loaders()
    
    