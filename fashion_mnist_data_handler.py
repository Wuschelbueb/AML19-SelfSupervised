"""
This module handles all FashionMNIST data related
functions like loading the data and the data loader.
"""

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST

from fashion_mnist_dataset import FashionMNISTDataset
from settings import TRANSFORM_FASHION_MNIST, BATCH_SIZE, ROOT_DIR_FASHION_MNIST


def train_data():
    """Download and load the training data."""
    return FashionMNIST(root=ROOT_DIR_FASHION_MNIST, download=True, train=True, transform=TRANSFORM_FASHION_MNIST)


def test_data():
    """Download and load the test data."""
    return FashionMNIST(root=ROOT_DIR_FASHION_MNIST, download=True, train=False, transform=TRANSFORM_FASHION_MNIST)


def train_loader_fashion_mnist():
    """Creates the data loader for the train data."""
    train_subset, _ = train_val_subset(0.8)

    train_set = FashionMNISTDataset(
        data=train_subset,
        targets=train_subset
    )

    return DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)


def val_loader_fashion_mnist():
    """Creates the data loader for the validation data."""
    _, val_subset = train_val_subset(0.8)

    val_set = FashionMNISTDataset(
        data=val_subset,
        targets=val_subset
    )

    return DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)


def test_loader_fashion_mnist():
    """Creates the data loader for the test data."""
    return DataLoader(test_data(), batch_size=BATCH_SIZE, shuffle=False)


def train_val_subset(split):
    """Splits the train data in train and validation subsets."""
    data_set = train_data()
    nbr_train_examples = int(len(data_set) * split)
    nbr_val_examples = len(data_set) - nbr_train_examples

    return random_split(data_set, [nbr_train_examples, nbr_val_examples])
