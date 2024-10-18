######################
# Dataset preparation
######################
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def get_data_loaders(batch_size: int = 64):
    """
    Downloads the FashionMNIST dataset and prepares DataLoader objects for training and testing.

    Args:
        batch_size (int): The size of batches to be used in training and evaluation. Default is 64.

    Returns:
        tuple: Contains DataLoader for train and test datasets.
    """
    # Define transformations for the dataset
    transform = ToTensor()

    # Downloading and loading train data
    train_data = FashionMNIST(
        root='./data',  # Assuming data should be stored in a 'data' folder in the current directory
        train=True,
        download=True,
        transform=transform
    )

    # Loading test data
    test_data = FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Creating DataLoader objects
    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True)

    test_dataloader = DataLoader(test_data,
                                 batch_size=batch_size,
                                 shuffle=False)

    return train_dataloader, test_dataloader