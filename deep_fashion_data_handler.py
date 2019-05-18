"""
This module handles all DeepFashion data related
functions like loading the data and the data loader.
"""

import pandas as pd
from torch.utils.data import DataLoader

from deep_fashion_dataset import DeepFashionDataset
from settings import ROOT_DIR_DEEP_FASHION, BATCH_SIZE


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
    partition = load_list_eval_partition()
    labels = load_list_category_img()

    result = partition[partition.evaluation_status == 'train']
    train_indices = result.index

    list_train_images = result.image_name.values
    train_labels = labels.category_label[train_indices].values

    return DeepFashionDataset(root=ROOT_DIR_DEEP_FASHION, image_list=list_train_images, targets=train_labels)


def val_data():
    """Return the images with the evaluation status 'val'."""
    partition = load_list_eval_partition()
    labels = load_list_category_img()

    result = partition[partition.evaluation_status == 'val']
    val_indices = result.index

    list_val_images = result.image_name.values
    val_labels = labels.category_label[val_indices].values

    return DeepFashionDataset(root=ROOT_DIR_DEEP_FASHION, image_list=list_val_images, targets=val_labels)


def test_data():
    """Return the images with the evaluation status 'test'."""
    partition = load_list_eval_partition()
    labels = load_list_category_img()

    result = partition[partition.evaluation_status == 'test']
    test_indices = result.index

    list_test_images = result.image_name.values
    test_labels = labels.category_label[test_indices].values

    return DeepFashionDataset(root=ROOT_DIR_DEEP_FASHION, image_list=list_test_images, targets=test_labels)


def train_loader_deep_fashion():
    """Return the data loader for the train data."""
    return DataLoader(train_data(), batch_size=BATCH_SIZE, shuffle=True)


def val_loader_deep_fashion():
    """Return the data loader for the validation data."""
    return DataLoader(val_data(), batch_size=BATCH_SIZE, shuffle=False)


def test_loader_deep_fashion():
    """Return the data loader for the test data."""
    return DataLoader(test_data(), batch_size=BATCH_SIZE, shuffle=False)
