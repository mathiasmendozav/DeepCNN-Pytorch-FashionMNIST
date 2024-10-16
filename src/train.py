#################
# Training Model
#################
from model import CNNModel
from dataset import get_data_loaders

if __name__ == '__main__':
    # Get data loaders
    train_dataloader, test_dataloader = get_data_loaders()
    
    
