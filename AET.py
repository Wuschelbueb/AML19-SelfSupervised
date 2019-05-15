import torch
import torch.nn as nn
import numpy as np
import time
from cifar_net import CifarNet
from decoder import Decoder
from deep_fashion_data_handler import train_loader_deep_fashion, val_loader_deep_fashion, test_loader_deep_fashion
from fashion_mnist_data_handler import train_loader_fashion_mnist, val_loader_fashion_mnist, test_loader_fashion_mnist
from fine_tune import fine_tune
from test import test

EPOCHS = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader_fashion_mnist = train_loader_fashion_mnist()
val_loader_fashion_mnist = val_loader_fashion_mnist()
test_loader_fashion_mnist = test_loader_fashion_mnist()

train_loader_deep_fashion = train_loader_deep_fashion()
val_loader_deep_fashion = val_loader_deep_fashion()
test_loader_deep_fashion = test_loader_deep_fashion()


class Classifier(nn.Module):
    'Convnet Classifier'

    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=train_loader_fashion_mnist.batch_size, out_channels=10, kernel_size=(4, 4), padding=0),
        )

    def forward(self, x):
        out = self.classifier(x)
        return out


def train_aet_cnn():
    """Trains the exemplar cnn model."""
    print("=============================================================")
    print("================ Train AET with FashionMNIST ================")
    print("=============================================================\n")

    # number of predicted classes = number of training images
    encoder = CifarNet(input_channels=1, num_classes=10)
    encoder = encoder.to(device)

    decoder = Decoder(input_channels=64, num_classes=10)
    decoder = decoder.to(device)

    parameters = list(encoder.parameters()) + list(decoder.parameters())

    # Criteria NLLLoss which is recommended with softmax final layer
    loss_fn = nn.MSELoss()

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(parameters, lr=0.001)

    # Decay LR by a factor of 0.1 every 4 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=4, gamma=0.1)

    return train(encoder, decoder, loss_fn, optimizer, scheduler, EPOCHS, train_loader_fashion_mnist)


def transfer_learning_aet(encoder):
    """Fine tunes the aet model."""
    print("=============================================================")
    print("=========== Transfer Learning with FashionMNIST =============")
    print("=============================================================\n")

    #classifier = Classifier()

    # Criteria NLLLoss which is recommended with Softmax final layer
    loss_fn = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 4 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=4, gamma=0.1)

    #classifier = classifier.to(device)
    encoder = encoder.to(device)

    return fine_tune(encoder, loss_fn, optimizer, scheduler, EPOCHS, train_loader_fashion_mnist, val_loader_fashion_mnist)


def test_aet(encoder):
    """Fine tunes the aet model."""
    print("=============================================================")
    print("============== Testing AET with FashionMNIST ================")
    print("=============================================================\n")

    encoder = encoder.to(device)

    loss_fn = nn.CrossEntropyLoss()

    return test(encoder, loss_fn, EPOCHS, test_loader_fashion_mnist)


def train(encoder, decoder, loss_fn, optimizer, scheduler, num_epochs, train_loader):
    """Train the model"""

    train_losses = []

    since = time.time()

    for epoch in range(num_epochs):
        scheduler.step()
        encoder.train()
        decoder.train()

        running_loss = []

        for images, labels in train_loader:
            images = images.to(device)

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

    return encoder, decoder, train_losses


# def fine_tune(encoder, loss_fn, optimizer, scheduler, num_epochs, train_loader):
#     """Train the model"""
#
#     train_losses = []
#
#     since = time.time()
#
#     for epoch in range(num_epochs):
#         scheduler.step()
#         encoder.train()
#
#         running_corrects = 0.0
#         running_loss = []
#
#         for images, labels in train_loader:
#             images = images.to(device)
#             labels = labels.to(device)
#
#             # zero the parameter gradients
#             optimizer.zero_grad()
#
#             output = encoder(images)
#
#             _, preds = torch.max(output.data, 1)
#
#             loss = loss_fn(output, labels)
#
#             # backward + optimize only if in training phase
#             loss.backward()
#             optimizer.step()
#
#             # statistics
#             running_loss.append(loss.item())
#
#             running_corrects += torch.sum(preds == labels.data).to(torch.float32)
#
#         train_losses.append(np.mean(np.array(running_loss)))
#
#         print('Epoch {}/{}: train_loss: {:.4f}'.format(
#             epoch + 1, num_epochs,
#             train_losses[-1]))
#
#     time_elapsed = time.time() - since
#
#     print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#     print('Average loss: {:4f}'.format(np.mean(train_losses)))
#
#     return encoder, train_losses


# def test(encoder, classifier, num_epochs, test_loader):
#     accuracies = []
#
#     for epoch in num_epochs:
#
#         correct_test_predictions = 0
#
#         for images, labels in test_loader:
#             images = images.to(device)
#             labels = labels.to(device)
#
#             output = encoder(images)
#             output = classifier(output)
#
#             predicted_label = output.argmax(dim=1)
#             n_correct = (predicted_label == labels).sum().item()
#             correct_test_predictions += n_correct
#
#         accuracy = (100.0*correct_test_predictions/len(test_loader_fashion_mnist.dataset))
#         print('Epoch {}/{}: accuracy: {:.4f}'.format(epoch + 1, num_epochs, accuracy))
#         accuracies.append(np.mean(accuracy))
#
#     return accuracies





