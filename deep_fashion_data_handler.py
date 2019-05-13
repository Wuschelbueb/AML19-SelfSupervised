"""
This module handles all DeepFashion data related
functions like loading the data and the data loader.
"""

import pandas as pd
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from deep_fashion_dataset import DeepFashionDataset

ROOT_DIR = 'img/'
TARGET_SIZE = (32, 32)


def load_images():
    """Load the images from the root directory."""
    transforms = Compose([
        Resize(TARGET_SIZE),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    return ImageFolder(ROOT_DIR, transform=transforms)


def load_list_category_img():
    """Load the list containing the image name and the associated category."""
    list_category_img = pd.read_csv('list_category_img.txt', sep="\t", header=0)
    list_category_img.columns = ["image_name", "category_label"]
    return list_category_img


def load_list_category_cloth():
    """Load the list containing the category name and the associated category type."""
    list_category_cloth = pd.read_csv('list_category_cloth.txt', sep="\t", header=0)
    list_category_cloth.columns = ["category_name", "category_type"]
    return list_category_cloth


def load_list_eval_partition():
    """Load the list containing the image name and the associated evaluation status."""
    list_eval_partition = pd.read_csv('list_eval_partition.txt', sep="\t", header=0)
    list_eval_partition.columns = ["image_name", "evaluation_status"]
    return list_eval_partition


def train_data():
    """Return the images with the evaluation status 'train'."""
    images = load_images()
    partition = load_list_eval_partition()
    labels = load_list_category_img()

    train_indices = partition[partition.evaluation_status == 'train'].index
    train_images = Subset(images, train_indices)
    train_labels = labels.category_label[train_indices].values

    return DeepFashionDataset(train_images, train_labels)


def val_data():
    """Return the images with the evaluation status 'val'."""
    images = load_images()
    partition = load_list_eval_partition()
    labels = load_list_category_img()

    val_indices = partition[partition.evaluation_status == 'val'].index
    val_images = Subset(images, val_indices)
    val_labels = labels.category_label[val_indices].values
    return DeepFashionDataset(val_images, val_labels)


def test_data():
    """Return the images with the evaluation status 'test'."""
    images = load_images()
    partition = load_list_eval_partition()
    labels = load_list_category_img()

    test_indices = partition[partition.evaluation_status == 'test'].index
    test_images = Subset(images, test_indices)
    test_labels = labels.category_label[test_indices].values
    return DeepFashionDataset(test_images, test_labels)


def train_loader_deep_fashion():
    """Return the data loader for the train data."""
    return DataLoader(train_data(), batch_size=64, shuffle=True)


def val_loader_deep_fashion():
    """Return the data loader for the validation data."""
    return DataLoader(val_data(), batch_size=64, shuffle=False)


def test_loader_deep_fashion():
    """Return the data loader for the test data."""
    return DataLoader(test_data(), batch_size=64, shuffle=False)
