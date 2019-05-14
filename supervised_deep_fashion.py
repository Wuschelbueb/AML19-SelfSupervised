import torch
import torch.nn as nn

from cifar_net import CifarNet
from fashion_mnist_data_handler import train_loader_classification, val_loader_classification, test_loader_classification
from deep_fashion_data_handler import train_loader_deep_fashion, val_loader_deep_fashion, test_loader_deep_fashion
from test import test
from train import train_and_val

EPOCHS = 15
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader_classification = train_loader_classification()
val_loader_classification = val_loader_classification()
test_loader_classification = test_loader_classification()

train_loader_deep_fashion = train_loader_deep_fashion()
val_loader_deep_fashion = val_loader_deep_fashion()
test_loader_deep_fashion = test_loader_deep_fashion()


def train_supervised_FashionMNIST():
    """Trains the supervised model."""
    print("=============================================================")
    print("============= Supervised Training FashionMNIST ==============")
    print("=============================================================\n")

    mnist_supervised_model = CifarNet(input_channels=1, num_classes=10)
    mnist_supervised_model = mnist_supervised_model.to(device)

    # Criteria NLLLoss which is recommended with Softmax final layer
    loss_fn = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(mnist_supervised_model.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 4 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=4, gamma=0.1)

    return train_and_val(mnist_supervised_model, loss_fn, optimizer, scheduler, EPOCHS,
                         train_loader_classification, val_loader_classification)


def train_supervised_deep_fashion():
    """Trains the supervised model."""
    print("============================================================")
    print("============= Supervised Training DeepFashion ==============")
    print("============================================================\n")

    df_supervised_model = CifarNet(input_channels=3, num_classes=50)
    df_supervised_model = df_supervised_model.to(device)

    # Criteria NLLLoss which is recommended with Softmax final layer
    loss_fn = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(df_supervised_model.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 4 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=4, gamma=0.1)

    return train_and_val(df_supervised_model, loss_fn, optimizer, scheduler, EPOCHS,
                         train_loader_deep_fashion, val_loader_deep_fashion)


def test_classification_on_supervised_fashionMNIST(model):
    """Tests the supervised model."""
    print("=============================================================")
    print("== Test Classification Supervised Model with FashionMNIST ===")
    print("=============================================================\n")

    # Criteria NLLLoss which is recommended with Softmax final layer
    loss_fn = nn.CrossEntropyLoss()

    model = model.to(device)
    return test(model, loss_fn, EPOCHS, test_loader_classification)


def test_classification_deep_fashion(model):
    """Fine tunes the rotation model."""
    print("============================================================")
    print("== Test Classification Supervised Model with DeepFashion ===")
    print("============================================================\n")

    # Criteria NLLLoss which is recommended with Softmax final layer
    loss_fn = nn.CrossEntropyLoss()

    model = model.to(device)
    return test(model, loss_fn, EPOCHS, test_loader_deep_fashion)
