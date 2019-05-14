"""Handles the rotation sub task."""
import torch
import torch.nn as nn

from cifar_net import CifarNet
from deep_fashion_data_handler import train_loader_deep_fashion, val_loader_deep_fashion, test_loader_deep_fashion, \
    train_loader_rotation_deep_fashion, val_loader_rotation_deep_fashion, test_loader_rotation_deep_fashion
from fashion_mnist_data_handler import train_loader_classification, val_loader_classification, \
    test_loader_classification, train_loader_rotation, \
    val_loader_rotation, test_loader_rotation
from test import test
from train import train_and_val

EPOCHS = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader_classification = train_loader_classification()
val_loader_classification = val_loader_classification()
test_loader_classification = test_loader_classification()

train_loader_rotation = train_loader_rotation()
val_loader_rotation = val_loader_rotation()
test_loader_rotation = test_loader_rotation()

train_loader_rotation_deep_fashion = train_loader_rotation_deep_fashion()
val_loader_rotation_deep_fashion = val_loader_rotation_deep_fashion()
test_loader_rotation_deep_fashion = test_loader_rotation_deep_fashion()


train_loader_deep_fashion = train_loader_deep_fashion()
val_loader_deep_fashion = val_loader_deep_fashion()
test_loader_deep_fashion = test_loader_deep_fashion()


def train_rotation_net():
    """Trains the rotation model."""
    print("=============================================================")
    print("========== Train Rotation Model with FashionMNIST ===========")
    print("=============================================================\n")

    rotation_model = CifarNet(input_channels=1, num_classes=4)
    rotation_model = rotation_model.to(device)

    # Criteria NLLLoss which is recommended with Softmax final layer
    loss_fn = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(rotation_model.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 4 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=4, gamma=0.1)

    return train_and_val(rotation_model, loss_fn, optimizer, scheduler, EPOCHS,
                         train_loader_rotation, val_loader_rotation)


def train_rotation_net_deep_fashion():
    """Trains the rotation model."""
    print("============================================================")
    print("========== Train Rotation Model with DeepFashion ===========")
    print("============================================================\n")

    rotation_model = CifarNet(input_channels=3, num_classes=4)
    rotation_model = rotation_model.to(device)

    # Criteria NLLLoss which is recommended with Softmax final layer
    loss_fn = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(rotation_model.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 4 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=4, gamma=0.1)

    return train_and_val(rotation_model, loss_fn, optimizer, scheduler, EPOCHS,
                         train_loader_rotation_deep_fashion, val_loader_rotation_deep_fashion)


def fine_tune_rotation_model(model):
    """Fine tunes the rotation model."""
    print("=============================================================")
    print("======== Fine Tune Rotation Model with FashionMNIST =========")
    print("=============================================================\n")

    # Criteria NLLLoss which is recommended with Softmax final layer
    loss_fn = nn.CrossEntropyLoss()

    # freezes all layers except the final one, according to the method parameters

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc3.parameters():
        param.requires_grad = True

    # replace fc layer with 10 outputs
    model.fc3 = nn.Sequential(nn.Linear(192, 192),
                              nn.Linear(192, 10, bias=True)
                              )

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 4 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=4, gamma=0.1)

    model = model.to(device)
    return train_and_val(model, loss_fn, optimizer, scheduler, EPOCHS,
                         train_loader_classification, val_loader_classification)


def fine_tune_rotation_model_deep_fashion(model):
    """Fine tunes the rotation model."""
    print("============================================================")
    print("======== Fine Tune Rotation Model with DeepFashion =========")
    print("============================================================\n")

    # Criteria NLLLoss which is recommended with Softmax final layer
    loss_fn = nn.CrossEntropyLoss()

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc3.parameters():
        param.requires_grad = True

    # replace fc layer with 10 outputs
    model.fc3 = nn.Sequential(nn.Linear(192, 192),
                              nn.Linear(192, 50, bias=True)
                              )

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 4 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=4, gamma=0.1)

    model = model.to(device)
    return train_and_val(model, loss_fn, optimizer, scheduler, EPOCHS,
                         train_loader_deep_fashion, val_loader_deep_fashion)


def test_classification_on_rotation_model(model):
    """Fine tunes the rotation model."""
    print("=============================================================")
    print("== Test Classification on Rotation Model with FashionMNIST ==")
    print("=============================================================\n")

    # Criteria NLLLoss which is recommended with Softmax final layer
    loss_fn = nn.CrossEntropyLoss()

    # replace fc layer with 10 outputs
    model.fc3 = nn.Sequential(nn.Linear(192, 192),
                              nn.Linear(192, 10, bias=True)
                              )

    model = model.to(device)
    return test(model, loss_fn, EPOCHS, test_loader_classification)


def test_classification_on_rotation_model_deep_fashion(model):
    """Fine tunes the rotation model."""
    print("============================================================")
    print("== Test Classification on Rotation Model with DeepFashion ==")
    print("============================================================\n")

    # Criteria NLLLoss which is recommended with Softmax final layer
    loss_fn = nn.CrossEntropyLoss()

    # replace fc layer with 10 outputs
    model.fc3 = nn.Sequential(nn.Linear(192, 192),
                              nn.Linear(192, 50, bias=True)
                              )

    model = model.to(device)
    return test(model, loss_fn, EPOCHS, test_loader_deep_fashion)
