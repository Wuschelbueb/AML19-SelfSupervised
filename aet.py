"""Handles the aet sub task."""
"""
is not working yet, has to be changed
is somewhat comparable to the aet/cifar10/projective/main.py
"""
import torch
import torch.nn as nn

from train import train_and_val
from test import test
from cifar_net import CifarNet
from fashion_mnist_data_handler import train_loader_classification, val_loader_classification, \
    test_loader_classification, train_loader_rotation, \
    val_loader_rotation, test_loader_rotation

EPOCHS = 15
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader_classification = train_loader_classification()
val_loader_classification = val_loader_classification()
test_loader_classification = test_loader_classification()

train_loader_rotation = train_loader_rotation()
val_loader_rotation = val_loader_rotation()
test_loader_rotation = test_loader_rotation()

def train_aet_net():
    """Trains the AET model."""
    print("===========================================")
    print("========== Train AET Model ===========")
    print("===========================================\n")
    #TODO: implement

def fine_tune_aet_model(model):
    """Fine tunes the AET model."""
    print("===========================================")
    print("======== Fine Tune AET Model =========")
    print("===========================================\n")
    #TODO: implement

def test_classification_on_aet_model(model):
    """Fine tunes the AET model."""
    print("===========================================")
    print("== Test Classification on AET Model ==")
    print("===========================================\n")
    #TODO: implement