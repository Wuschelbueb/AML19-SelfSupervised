from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from FashionMNISTDataset import FashionMNISTDataset
from FashionMNISTRotation import FashionMNISTRotation
from exemplarCNN import ExemplarCNN

target_size = 32
transform = Compose([Resize(target_size), ToTensor(), Normalize(mean=(0.5,), std=(0.5,))])
batch_size = 64
root_dir = 'fashion_mnist'


def train_data():
    """Download and load the training data."""
    print("Load train data")
    return FashionMNIST(root=root_dir, download=True, train=True, transform=transform)


def test_data():
    """Download and load the test data."""
    print("Load test data")
    return FashionMNIST(root=root_dir, download=True, train=False, transform=transform)


def train_loader_classification():
    train_subset, val_subset = train_val_subset(0.8)

    train_set = FashionMNISTDataset(
        data=train_subset,
        targets=train_subset
    )

    return DataLoader(train_set, batch_size=64, shuffle=True)


def val_loader_classification():
    train_subset, val_subset = train_val_subset(0.8)

    val_set = FashionMNISTDataset(
        data=val_subset,
        targets=val_subset
    )

    return DataLoader(val_set, batch_size=64, shuffle=False)


def test_loader_classification():
    """Get the test DataLoader"""
    return DataLoader(test_data(), batch_size=batch_size, shuffle=False)


def train_data_rotation():
    train_sub, _ = train_val_subset(0.8)

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
    print('Size of train set for rotation: {}'.format(len(train_data_rotation)))
    return train_data_rotation


def train_loader_rotation():
    return DataLoader(train_data_rotation(), batch_size=batch_size, shuffle=True, num_workers=1)


def val_data_rotation():
    _, val_subset = train_val_subset(0.8)

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
    print('Size of validation set for rotation: {}'.format(len(val_data_rotation)))
    return val_data_rotation


def val_loader_rotation():
    return DataLoader(val_data_rotation(), batch_size=batch_size, shuffle=False, num_workers=1)


def test_data_rotation():
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

    test_data_set_rotation = ConcatDataset([test_set_0, test_set_90, test_set_180, test_set_270])
    print('Size of train set for rotation: {}'.format(len(test_data_set_rotation)))
    return test_data_set_rotation


def test_loader_rotation():
    return DataLoader(test_data_rotation(), batch_size=batch_size, shuffle=False, num_workers=1)


def train_data_exemplar_cnn():
    train_subset, val_subset = train_val_subset(0.5)

    train_set_exemplar_cnn = ExemplarCNN(
        data=train_subset,
        target=train_subset,
    )

    print('Size of train set for exemplar cnn: {}'.format(len(train_set_exemplar_cnn)))
    return train_set_exemplar_cnn


def val_data_exemplar_cnn():
    train_subset, val_subset = train_val_subset(0.5)

    val_set_exemplar_cnn = ExemplarCNN(
        data=val_subset,
        target=val_subset,
    )

    print('Size of validation set for exemplar cnn: {}'.format(len(val_set_exemplar_cnn)))
    return val_set_exemplar_cnn


def test_data_exemplar_cnn():
    data = test_data()

    test_set_exemplar_cnn = ExemplarCNN(
        data=data,
        target=data,
    )

    print('Size of test set for exemplar cnn: {}'.format(len(test_set_exemplar_cnn)))
    return test_set_exemplar_cnn


def train_loader_exemplar_cnn():
    return DataLoader(train_data_exemplar_cnn(), batch_size=64, shuffle=True)


def val_loader_exemplar_cnn():
    return DataLoader(val_data_exemplar_cnn(), batch_size=64, shuffle=False)


def test_loader_exemplar_cnn():
    return DataLoader(test_data_exemplar_cnn(), batch_size=64, shuffle=False)


def train_val_subset(split):
    data_set = train_data()
    nbr_train_examples = int(len(data_set) * split)
    nbr_val_examples = len(data_set) - nbr_train_examples

    return random_split(data_set, [nbr_train_examples, nbr_val_examples])
