"""
This module handles all FashionMNIST data related
functions like loading the data and the data loader.
"""

from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from fashion_mnist_dataset import FashionMNISTDataset
from fashion_mnist_rotation_dataset import FashionMNISTRotationDataset
from fashion_mnist_exemplar_cnn_dataset import FashionMNISTExemplarCNNDataset

BATCH_SIZE = 64
ROOT_DIR = 'fashion_mnist'
TARGET_SIZE = 32
TRANSFORM = Compose([Resize(TARGET_SIZE), ToTensor(), Normalize(mean=(0.5,), std=(0.5,))])


def train_data():
    """Download and load the training data."""
    return FashionMNIST(root=ROOT_DIR, download=True, train=True, transform=TRANSFORM)


def test_data():
    """Download and load the test data."""
    return FashionMNIST(root=ROOT_DIR, download=True, train=False, transform=TRANSFORM)


def train_loader_classification():
    """Creates the data loader for the train data."""
    train_subset, _ = train_val_subset(0.8)

    train_set = FashionMNISTDataset(
        data=train_subset,
        targets=train_subset
    )

    return DataLoader(train_set, batch_size=64, shuffle=True)


def val_loader_classification():
    """Creates the data loader for the validation data."""
    _, val_subset = train_val_subset(0.8)

    val_set = FashionMNISTDataset(
        data=val_subset,
        targets=val_subset
    )

    return DataLoader(val_set, batch_size=64, shuffle=False)


def test_loader_classification():
    """Creates the data loader for the test data."""
    return DataLoader(test_data(), batch_size=BATCH_SIZE, shuffle=False)


def train_data_rotation():
    """Creates the train data for the rotation task."""
    train_sub, _ = train_val_subset(0.8)

    train_set_0 = FashionMNISTRotationDataset(
        data=train_sub,
        target=train_sub,
        angle=0
    )

    train_set_90 = FashionMNISTRotationDataset(
        data=train_sub,
        target=train_sub,
        angle=90
    )

    train_set_180 = FashionMNISTRotationDataset(
        data=train_sub,
        target=train_sub,
        angle=180
    )

    train_set_270 = FashionMNISTRotationDataset(
        data=train_sub,
        target=train_sub,
        angle=270
    )

    train_set_rotation = ConcatDataset([train_set_0, train_set_90, train_set_180, train_set_270])
    print('Size of train set for rotation: {}'.format(len(train_set_rotation)))
    return train_set_rotation


def train_loader_rotation():
    """Creates the data loader for the rotation train data."""
    return DataLoader(train_data_rotation(), batch_size=BATCH_SIZE, shuffle=True, num_workers=1)


def val_data_rotation():
    """Creates the validation data for the rotation task."""
    _, val_subset = train_val_subset(0.8)

    val_set_0 = FashionMNISTRotationDataset(
        data=val_subset,
        target=val_subset,
        angle=0
    )

    val_set_90 = FashionMNISTRotationDataset(
        data=val_subset,
        target=val_subset,
        angle=90
    )

    val_set_180 = FashionMNISTRotationDataset(
        data=val_subset,
        target=val_subset,
        angle=180
    )

    val_set_270 = FashionMNISTRotationDataset(
        data=val_subset,
        target=val_subset,
        angle=270
    )

    val_set_rotation = ConcatDataset([val_set_0, val_set_90, val_set_180, val_set_270])
    print('Size of validation set for rotation: {}'.format(len(val_set_rotation)))
    return val_set_rotation


def val_loader_rotation():
    """Creates the data loader for the rotation validation data."""
    return DataLoader(val_data_rotation(), batch_size=BATCH_SIZE, shuffle=False, num_workers=1)


def test_data_rotation():
    """Creates the test data for the rotation task."""
    data_set = test_data()

    test_set_0 = FashionMNISTRotationDataset(
        data=data_set,
        target=data_set,
        angle=0
    )

    test_set_90 = FashionMNISTRotationDataset(
        data=data_set,
        target=data_set,
        angle=90
    )

    test_set_180 = FashionMNISTRotationDataset(
        data=data_set,
        target=data_set,
        angle=180
    )

    test_set_270 = FashionMNISTRotationDataset(
        data=data_set,
        target=data_set,
        angle=270
    )

    test_data_set_rotation = ConcatDataset([test_set_0, test_set_90, test_set_180, test_set_270])
    print('Size of train set for rotation: {}'.format(len(test_data_set_rotation)))
    return test_data_set_rotation


def test_loader_rotation():
    """Creates the data loader for the rotation test data."""
    return DataLoader(test_data_rotation(), batch_size=BATCH_SIZE, shuffle=False, num_workers=1)


def train_data_exemplar_cnn():
    """Creates the train data for the exemplar cnn task."""
    data = train_data()

    train_set_exemplar_cnn = FashionMNISTExemplarCNNDataset(
        data=data,
        target=data,
    )

    print('Size of train set for exemplar cnn: {}'.format(len(train_set_exemplar_cnn)))
    return train_set_exemplar_cnn


def test_data_exemplar_cnn():
    """Creates the test data for the rotation task."""
    data = test_data()

    test_set_exemplar_cnn = FashionMNISTExemplarCNNDataset(
        data=data,
        target=data,
    )

    print('Size of test set for exemplar cnn: {}'.format(len(test_set_exemplar_cnn)))
    return test_set_exemplar_cnn


def train_loader_exemplar_cnn():
    """Creates the data loader for the exemplar cnn train data."""
    return DataLoader(train_data_exemplar_cnn(), batch_size=64, shuffle=False)


def test_loader_exemplar_cnn():
    """Creates the data loader for the exemplar cnn test data."""
    return DataLoader(test_data_exemplar_cnn(), batch_size=64, shuffle=False)


def train_val_subset(split):
    """Splits the train data in train and validation subsets."""
    data_set = train_data()
    nbr_train_examples = int(len(data_set) * split)
    nbr_val_examples = len(data_set) - nbr_train_examples

    return random_split(data_set, [nbr_train_examples, nbr_val_examples])
