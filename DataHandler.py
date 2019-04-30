from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, ToTensor, Normalize

from FashionMNISTDataset import FashionMNISTDataset
from FashionMNISTRotation import FashionMNISTRotation

transform = Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))])
batch_size = 64
root_dir = 'fashion_mnist'


def train_data():
    """Download and load the training data."""
    return FashionMNIST(root=root_dir, download=True, train=True, transform=transform)


def test_data():
    """Download and load the test data."""
    return FashionMNIST(root=root_dir, download=True, train=False, transform=transform)


def train_data_loader():
    """Get the train DataLoader"""
    return DataLoader(train_data(), batch_size=batch_size, shuffle=True)


def test_data_loader():
    """Get the test DataLoader"""
    return DataLoader(test_data(), batch_size=batch_size, shuffle=False)


def train_val_dataloader():
    train_subset, val_subset = train_val_subset()

    train_set = FashionMNISTDataset(
        data=train_subset,
        targets=train_subset
    )

    val_set = FashionMNISTDataset(
        data=val_subset,
        targets=val_subset
    )

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

    return train_loader, val_loader


def test_data_rotation_loader():
    data_set = test_data()

    test_set_0 = FashionMNISTRotation(
        data=data_set,
        target=data_set,
        angle=0
    )

    test_set_90 = FashionMNISTRotation(
        data=data_set,
        target=data_set,
        angle=90
    )

    test_set_180 = FashionMNISTRotation(
        data=data_set,
        target=data_set,
        angle=180
    )

    test_set_270 = FashionMNISTRotation(
        data=data_set,
        target=data_set,
        angle=270
    )

    test_data_rotation = ConcatDataset([test_set_0, test_set_90, test_set_180, test_set_270])
    test_loader_rotation = DataLoader(test_data_rotation, batch_size=batch_size, shuffle=False, num_workers=1)

    return test_loader_rotation


def train_data_rotation_loader():
    train_sub, _ = train_val_subset()

    train_set_0 = FashionMNISTRotation(
        data=train_sub,
        target=train_sub,
        angle=0
    )

    train_set_90 = FashionMNISTRotation(
        data=train_sub,
        target=train_sub,
        angle=90
    )

    train_set_180 = FashionMNISTRotation(
        data=train_sub,
        target=train_sub,
        angle=180
    )

    train_set_270 = FashionMNISTRotation(
        data=train_sub,
        target=train_sub,
        angle=270
    )

    train_data_rotation = ConcatDataset([train_set_0, train_set_90, train_set_180, train_set_270])
    train_loader_rotation = DataLoader(train_data_rotation, batch_size=batch_size, shuffle=True, num_workers=1)

    return train_loader_rotation


def val_data_rotation_loader():
    _, val_subset = train_val_subset()

    val_set_0 = FashionMNISTRotation(
        data=val_subset,
        target=val_subset,
        angle=0
    )

    val_set_90 = FashionMNISTRotation(
        data=val_subset,
        target=val_subset,
        angle=90
    )

    val_set_180 = FashionMNISTRotation(
        data=val_subset,
        target=val_subset,
        angle=180
    )

    val_set_270 = FashionMNISTRotation(
        data=val_subset,
        target=val_subset,
        angle=270
    )

    val_data_rotation = ConcatDataset([val_set_0, val_set_90, val_set_180, val_set_270])
    val_loader_rotation = DataLoader(val_data_rotation, batch_size=batch_size, shuffle=False, num_workers=1)

    return val_loader_rotation


def train_val_subset():
    data_set = train_data()
    nbr_train_examples = int(len(data_set) * 0.8)
    nbr_val_examples = len(data_set) - nbr_train_examples

    return random_split(data_set, [nbr_train_examples, nbr_val_examples])

