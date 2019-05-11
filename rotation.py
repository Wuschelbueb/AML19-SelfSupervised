"""Handles the rotation sub task."""
import torch
import torch.nn as nn

from train import train_and_val
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


def train_rotation_net():
    """Trains the rotation model."""
    print("===========================================")
    print("========== Train Rotation Model ===========")
    print("===========================================\n")

    # rotation_model = ResNet20()
    rotation_model = CifarNet(input_channels=1, num_classes=4)
    rotation_model = rotation_model.to(device)

    # fitting the convolution to 1 input channel (instead of 3)
    rotation_model.conv = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)

    # Criteria NLLLoss which is recommended with Softmax final layer
    loss_fn = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(rotation_model.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 4 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=4, gamma=0.1)

    return train_and_val(rotation_model, loss_fn, optimizer, scheduler, EPOCHS,
                         train_loader_rotation, val_loader_rotation)


def fine_tune_rotation_model(model, unfreeze_l1, unfreeze_l2, unfreeze_l3, unfreeze_fc):
    """Fine tunes the rotation model."""
    print("===========================================")
    print("======== Fine Tune Rotation Model =========")
    print("===========================================\n")

    # Criteria NLLLoss which is recommended with Softmax final layer
    loss_fn = nn.CrossEntropyLoss()

    # freezes the layers according to the method parameters
    for param in model.layer1.parameters():
        param.requires_grad = unfreeze_l1

    for param in model.layer2.parameters():
        param.requires_grad = unfreeze_l2

    for param in model.layer3.parameters():
        param.requires_grad = unfreeze_l3

    for param in model.fc.parameters():
        param.requires_grad = unfreeze_fc

    # replace fc layer with 10 outputs
    model.fc = nn.Linear(64, 10)

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 4 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=4, gamma=0.1)

    model = model.to(device)
    return train_and_val(model, loss_fn, optimizer, scheduler, EPOCHS,
                         train_loader_classification, val_loader_classification)
