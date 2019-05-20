import torch
import torch.nn as nn

from cifar_net import CifarNet
from deep_fashion_data_handler import train_loader_deep_fashion, val_loader_deep_fashion, test_loader_deep_fashion
from fashion_mnist_data_handler import train_loader_fashion_mnist, val_loader_fashion_mnist, test_loader_fashion_mnist
from fine_tune import fine_tune
from settings import DEVICE, EPOCHS, STEP_SIZE_TRAIN, GAMMA, LEARNING_RATE_TRAIN
from test import test

train_loader_fashion_mnist = train_loader_fashion_mnist()
val_loader_fashion_mnist = val_loader_fashion_mnist()
test_loader_fashion_mnist = test_loader_fashion_mnist()

train_loader_deep_fashion = train_loader_deep_fashion()
val_loader_deep_fashion = val_loader_deep_fashion()
test_loader_deep_fashion = test_loader_deep_fashion()


def train_supervised_FashionMNIST():
    """Trains the supervised model."""
    print("=============================================================")
    print("============= Supervised Training FashionMNIST ==============")
    print("=============================================================\n")

    mnist_supervised_model = CifarNet(input_channels=1, num_classes=10)
    mnist_supervised_model = mnist_supervised_model.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(mnist_supervised_model.parameters(), lr=LEARNING_RATE_TRAIN)

    # Decay LR by a factor of GAMMA
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=STEP_SIZE_TRAIN, gamma=GAMMA)

    return fine_tune(mnist_supervised_model, loss_fn, optimizer, scheduler, EPOCHS,
                     train_loader_fashion_mnist, val_loader_fashion_mnist)


def train_supervised_deep_fashion():
    """Trains the supervised model."""
    print("============================================================")
    print("============= Supervised Training DeepFashion ==============")
    print("============================================================\n")

    df_supervised_model = CifarNet(input_channels=3, num_classes=50)
    df_supervised_model = df_supervised_model.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(df_supervised_model.parameters(), lr=LEARNING_RATE_TRAIN)

    # Decay LR by a factor of GAMMA
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=STEP_SIZE_TRAIN, gamma=GAMMA)

    return fine_tune(df_supervised_model, loss_fn, optimizer, scheduler, EPOCHS,
                     train_loader_deep_fashion, val_loader_deep_fashion)


def test_classification_on_supervised_fashionMNIST(model):
    """Tests the supervised model."""
    print("=============================================================")
    print("== Test Classification Supervised Model with FashionMNIST ===")
    print("=============================================================\n")

    loss_fn = nn.CrossEntropyLoss()
    model = model.to(DEVICE)
    return test(model, loss_fn, test_loader_fashion_mnist)


def test_classification_deep_fashion(model):
    """Tests the supervised model."""
    print("============================================================")
    print("== Test Classification Supervised Model with DeepFashion ===")
    print("============================================================\n")

    loss_fn = nn.CrossEntropyLoss()
    model = model.to(DEVICE)
    return test(model, loss_fn, test_loader_deep_fashion)
