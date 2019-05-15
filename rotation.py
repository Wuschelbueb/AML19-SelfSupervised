"""Handles the rotation sub task."""

import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as tf

from cifar_net import CifarNet
from deep_fashion_data_handler import train_loader_deep_fashion, val_loader_deep_fashion, test_loader_deep_fashion
from fashion_mnist_data_handler import train_loader_fashion_mnist, val_loader_fashion_mnist, test_loader_fashion_mnist
from fine_tune import fine_tune
from settings import DEVICE, EPOCHS
from test import test

train_loader_fashion_mnist = train_loader_fashion_mnist()
val_loader_fashion_mnist = val_loader_fashion_mnist()
test_loader_fashion_mnist = test_loader_fashion_mnist()

train_loader_deep_fashion = train_loader_deep_fashion()
val_loader_deep_fashion = val_loader_deep_fashion()
test_loader_deep_fashion = test_loader_deep_fashion()


def rotate(image, angle):
    """Rotate the image by the specified angle"""
    image = image.cpu()
    image = tf.to_pil_image(image)
    image = tf.rotate(image, angle)
    image = tf.to_tensor(image)
    if image.shape[0] == 3:
        image = tf.normalize(image, (0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
    else:
        image = tf.normalize(image, (0.5,), (0.5,))
    image = image.cuda()
    return image


def train_rotation_net():
    """Trains the rotation model."""
    print("=============================================================")
    print("========== Train Rotation Model with FashionMNIST ===========")
    print("=============================================================\n")

    model = CifarNet(input_channels=1, num_classes=4)
    model = model.to(DEVICE)

    # Criteria NLLLoss which is recommended with Softmax final layer
    loss_fn = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 4 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=4, gamma=0.1)

    return train(model, loss_fn, optimizer, scheduler, EPOCHS, train_loader_fashion_mnist, val_loader_fashion_mnist)


def train_rotation_net_deep_fashion():
    """Trains the rotation model."""
    print("============================================================")
    print("========== Train Rotation Model with DeepFashion ===========")
    print("============================================================\n")

    model = CifarNet(input_channels=3, num_classes=4)
    model = model.to(DEVICE)

    # Criteria NLLLoss which is recommended with Softmax final layer
    loss_fn = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 4 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=4, gamma=0.1)

    return train(model, loss_fn, optimizer, scheduler, EPOCHS, train_loader_deep_fashion, val_loader_deep_fashion)


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

    model = model.to(DEVICE)
    return fine_tune(model, loss_fn, optimizer, scheduler, EPOCHS, train_loader_fashion_mnist, val_loader_fashion_mnist)


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

    # replace fc layer with 50 outputs
    model.fc3 = nn.Sequential(nn.Linear(192, 192),
                              nn.Linear(192, 50, bias=True)
                              )

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 4 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=4, gamma=0.1)

    model = model.to(DEVICE)
    return fine_tune(model, loss_fn, optimizer, scheduler, EPOCHS, train_loader_deep_fashion, val_loader_deep_fashion)


def test_classification_on_rotation_model(model):
    """Tests the rotation model."""
    print("=============================================================")
    print("== Test Classification on Rotation Model with FashionMNIST ==")
    print("=============================================================\n")

    # Criteria NLLLoss which is recommended with Softmax final layer
    loss_fn = nn.CrossEntropyLoss()

    # replace fc layer with 10 outputs
    model.fc3 = nn.Sequential(nn.Linear(192, 192),
                              nn.Linear(192, 10, bias=True)
                              )

    model = model.to(DEVICE)
    return test(model, loss_fn, test_loader_fashion_mnist)


def test_classification_on_rotation_model_deep_fashion(model):
    """Tests the rotation model."""
    print("============================================================")
    print("== Test Classification on Rotation Model with DeepFashion ==")
    print("============================================================\n")

    # Criteria NLLLoss which is recommended with Softmax final layer
    loss_fn = nn.CrossEntropyLoss()

    # replace fc layer with 50 outputs
    model.fc3 = nn.Sequential(nn.Linear(192, 192),
                              nn.Linear(192, 50, bias=True)
                              )

    model = model.to(DEVICE)
    return test(model, loss_fn, test_loader_deep_fashion)


def train(model, loss_fn, optimizer, scheduler, num_epochs, train_loader, val_loader):
    """Train the model"""
    # We will monitor loss functions as the training progresses
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_model_wts = model.state_dict()
    best_acc = 0.0

    since = time.time()

    for epoch in range(num_epochs):
        scheduler.step()
        model.train()

        running_loss = []
        running_corrects_train = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            images_rotated = []
            labes_rotated = []

            for img in images:
                rotated_imgs = [
                    img,
                    rotate(img, 90),
                    rotate(img, 180),
                    rotate(img, 270)
                ]
                rotation_labels = torch.LongTensor([0, 1, 2, 3])
                stack = torch.stack(rotated_imgs, dim=0)

                images_rotated.append(stack)
                labes_rotated.append(rotation_labels)

            images = torch.cat(images_rotated, dim=0).to(DEVICE)
            labels = torch.cat(labes_rotated, dim=0).to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            loss = loss_fn(outputs, labels)

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

            # statistics
            running_loss.append(loss.item())
            running_corrects_train += torch.sum(preds == labels.data).to(torch.float32)

        train_losses.append(np.mean(np.array(running_loss)))
        train_accuracies.append((100.0 * running_corrects_train) / (4 * len(train_loader.dataset)))

        model.eval()
        running_corrects_val = 0.0
        running_loss = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                images_rotated = []
                labes_rotated = []

                for img in images:
                    rotated_imgs = [
                        img,
                        rotate(img, 90),
                        rotate(img, 180),
                        rotate(img, 270)
                    ]
                    rotation_labels = torch.LongTensor([0, 1, 2, 3])
                    stack = torch.stack(rotated_imgs, dim=0)

                    images_rotated.append(stack)
                    labes_rotated.append(rotation_labels)

                images = torch.cat(images_rotated, dim=0).to(DEVICE)
                labels = torch.cat(labes_rotated, dim=0).to(DEVICE)

                # forward
                outputs = model(images)
                _, preds = torch.max(outputs.data, 1)
                loss = loss_fn(outputs, labels)

                # statistics
                running_loss.append(loss.item())
                running_corrects_val += torch.sum(preds == labels.data).to(torch.float32)

        val_losses.append(np.mean(np.array(running_loss)))
        val_accuracies.append((100.0 * running_corrects_val) / (4 * len(val_loader.dataset)))

        if val_accuracies[-1] > best_acc:
            best_acc = val_accuracies[-1]

        print('Epoch {}/{}: train_loss: {:.4f}, train_accuracy: {:.4f}, val_loss: {:.4f}, val_accuracy: {:.4f}'.format(
            epoch + 1, num_epochs,
            train_losses[-1],
            train_accuracies[-1],
            val_losses[-1],
            val_accuracies[-1]))

    time_elapsed = time.time() - since
    model.load_state_dict(best_model_wts)  # load best model weights

    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model, train_losses, val_losses, train_accuracies, val_accuracies
