import time

import numpy as np
import torch
import torch.nn as nn

from cifar_net import CifarNet
from decoder import Decoder
from deep_fashion_data_handler import train_loader_deep_fashion, val_loader_deep_fashion, test_loader_deep_fashion
from fashion_mnist_data_handler import train_loader_fashion_mnist, val_loader_fashion_mnist, test_loader_fashion_mnist
from settings import DEVICE, EPOCHS, STEP_SIZE_TRAIN, GAMMA, LEARNING_RATE_TRAIN, LEARNING_RATE_FINE_TUNE, WEIGHT_DECAY, \
    STEP_SIZE_FINE_TUNE, EPOCHS_FINE_TUNE

train_loader_fashion_mnist = train_loader_fashion_mnist()
val_loader_fashion_mnist = val_loader_fashion_mnist()
test_loader_fashion_mnist = test_loader_fashion_mnist()

train_loader_deep_fashion = train_loader_deep_fashion()
val_loader_deep_fashion = val_loader_deep_fashion()
test_loader_deep_fashion = test_loader_deep_fashion()


def train_autoencoder_mnist():
    """Trains the autoencoder for Fashion MNIST."""
    print("=============================================================")
    print("================ Train AE with FashionMNIST =================")
    print("=============================================================\n")

    encoder = CifarNet(input_channels=1, num_classes=10)
    encoder = encoder.to(DEVICE)

    decoder = Decoder(input_channels=64, num_classes=10, out_channels=1)
    decoder = decoder.to(DEVICE)

    parameters = list(encoder.parameters()) + list(decoder.parameters())
    loss_fn = nn.MSELoss()

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE_TRAIN)

    # Decay LR by a factor of GAMMA
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=STEP_SIZE_TRAIN, gamma=GAMMA)

    return train(encoder, decoder, loss_fn, optimizer, scheduler, EPOCHS, train_loader_fashion_mnist)


def train_autoencoder_deep_fashion():
    """Trains the autoencoder for DeepFashion."""
    print("=============================================================")
    print("================ Train AE with DeepFashion ==================")
    print("=============================================================\n")

    encoder = CifarNet(input_channels=3, num_classes=50)
    encoder = encoder.to(DEVICE)

    decoder = Decoder(input_channels=64, num_classes=50, out_channels=3)
    decoder = decoder.to(DEVICE)

    parameters = list(encoder.parameters()) + list(decoder.parameters())
    loss_fn = nn.MSELoss()

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE_TRAIN)

    # Decay LR by a factor of GAMMA
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=STEP_SIZE_TRAIN, gamma=GAMMA)

    return train(encoder, decoder, loss_fn, optimizer, scheduler, EPOCHS, train_loader_deep_fashion)


def fine_tune_autoencoder_mnist(encoder):
    """Fine tunes the autoencoder for Fashion MNIST."""
    print("=============================================================")
    print("=========== Transfer Learning with FashionMNIST =============")
    print("=============================================================\n")

    encoder = encoder.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    parameters = list(encoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE_FINE_TUNE, weight_decay=WEIGHT_DECAY)

    # Decay LR by a factor of GAMMA
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=STEP_SIZE_FINE_TUNE, gamma=GAMMA)

    return fine_tune_autoencoder(encoder, loss_fn, optimizer, scheduler, EPOCHS_FINE_TUNE,
                                 train_loader_fashion_mnist, val_loader_fashion_mnist)


def fine_tune_autoencoder_deep_fashion(encoder):
    """Fine tunes the autoencoder for DeepFashion."""
    print("=============================================================")
    print("=========== Transfer Learning with DeepFashion ==============")
    print("=============================================================\n")

    encoder = encoder.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    parameters = list(encoder.parameters())
    optimizer = torch.optim.Adam(parameters,  lr=LEARNING_RATE_FINE_TUNE, weight_decay=WEIGHT_DECAY)

    # Decay LR by a factor of GAMMA
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=STEP_SIZE_FINE_TUNE, gamma=GAMMA)

    return fine_tune_autoencoder(encoder, loss_fn, optimizer, scheduler, EPOCHS_FINE_TUNE,
                                 train_loader_deep_fashion, val_loader_deep_fashion)


def test_autoencoder_mnist(encoder):
    """Tests the autoencoder."""
    print("=============================================================")
    print("============== Testing AE with FashionMNIST  ================")
    print("=============================================================\n")

    encoder = encoder.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    return test_autoencoder(encoder, loss_fn, test_loader_fashion_mnist)


def test_autoencoder_deep_fashion(encoder):
    """Tests the autoencoder."""
    print("=============================================================")
    print("============== Testing AE with FashionMNIST =================")
    print("=============================================================\n")

    encoder = encoder.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    return test_autoencoder(encoder, loss_fn, test_loader_deep_fashion)


def train(encoder, decoder, loss_fn, optimizer, scheduler, num_epochs, train_loader):
    """Train the model"""

    since = time.time()
    train_losses = []

    for epoch in range(num_epochs):
        scheduler.step()
        encoder.train()
        decoder.train()

        running_loss = []

        for images, labels in train_loader:
            images = images.to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            output = encoder(images)
            output = decoder(output)

            loss = loss_fn(output, images)

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

            # statistics
            running_loss.append(loss.item())

        train_losses.append(np.mean(np.array(running_loss)))

        print('Epoch {}/{}: train_loss: {:.4f}'.format(
            epoch + 1, num_epochs,
            train_losses[-1]))

    time_elapsed = time.time() - since

    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Average loss: {:4f}'.format(np.mean(train_losses)))

    return encoder, train_losses


def fine_tune_autoencoder(encoder, loss_fn, optimizer, scheduler, num_epochs, train_loader, val_loader):
    """Fine tune the model"""
    since = time.time()

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_acc = 0.0
    best_model_wts = encoder.state_dict()

    for epoch in range(num_epochs):
        scheduler.step()
        encoder.train()

        running_loss = []
        running_corrects_train = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = encoder(images)

            _, preds = torch.max(outputs.data, 1)
            loss = loss_fn(outputs, labels)

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

            # statistics
            running_loss.append(loss.item())
            running_corrects_train += torch.sum(preds == labels.data).to(torch.float32)

        train_losses.append(np.mean(np.array(running_loss)))
        train_accuracies.append(100.0 * running_corrects_train / len(train_loader.dataset))

        encoder.eval()
        running_loss = []
        running_corrects_val = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                # forward
                outputs = encoder(images)

                _, preds = torch.max(outputs.data, 1)
                loss = loss_fn(outputs, labels)

                # statistics
                running_loss.append(loss.item())
                running_corrects_val += torch.sum(preds == labels.data).to(torch.float32)

        val_losses.append(np.mean(np.array(running_loss)))
        val_accuracies.append(100.0 * running_corrects_val / len(val_loader.dataset))

        if val_accuracies[-1] > best_acc:
            best_acc = val_accuracies[-1]
            best_model_wts = encoder.state_dict()

        print('Epoch {}/{}: train_loss: {:.4f}, train_accuracy: {:.4f}, val_loss: {:.4f}, val_accuracy: {:.4f}'.format(
            epoch + 1, num_epochs,
            train_losses[-1],
            train_accuracies[-1],
            val_losses[-1],
            val_accuracies[-1]))

    time_elapsed = time.time() - since
    encoder.load_state_dict(best_model_wts)  # load best model weights

    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return encoder, train_losses, val_losses, train_accuracies, val_accuracies


def test_autoencoder(encoder, loss_fn, test_loader):
    """Tests the model on data from test_loader"""
    encoder.eval()
    test_loss = 0.0
    n_correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = encoder(images)

            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
            n_correct += torch.sum(outputs.argmax(1) == labels).item()

    average_test_loss = test_loss / len(test_loader.dataset)
    average_test_accuracy = 100.0 * n_correct / len(test_loader.dataset)

    print('Test average loss:', average_test_loss, 'accuracy:', average_test_accuracy)
    return average_test_loss, average_test_accuracy
