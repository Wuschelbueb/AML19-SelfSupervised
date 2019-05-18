"""Contains the code for the exemplar cnn sub task."""
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, ColorJitter, \
    RandomResizedCrop, RandomRotation, RandomAffine, Compose, Resize, ToTensor, \
    Normalize, ToPILImage

from cifar_net import CifarNet
from deep_fashion_data_handler import train_loader_deep_fashion, val_loader_deep_fashion, test_loader_deep_fashion
from fashion_mnist_data_handler import train_loader_fashion_mnist, val_loader_fashion_mnist, test_loader_fashion_mnist
from fine_tune import fine_tune
from settings import DEVICE, EPOCHS, LEARNING_RATE_TRAIN, WEIGHT_DECAY, STEP_SIZE_TRAIN, GAMMA, LEARNING_RATE_FINE_TUNE, \
    STEP_SIZE_FINE_TUNE, EPOCHS_FINE_TUNE
from test import test

train_loader_fashion_mnist = train_loader_fashion_mnist()
val_loader_fashion_mnist = val_loader_fashion_mnist()
test_loader_fashion_mnist = test_loader_fashion_mnist()

train_loader_deep_fashion = train_loader_deep_fashion()
val_loader_deep_fashion = val_loader_deep_fashion()
test_loader_deep_fashion = test_loader_deep_fashion()


def transform_image(image, transformation):
    """Randomly transforms one image."""
    image = image.cpu()
    transform = ToPILImage()
    img = transform(image)

    if transformation == 0:
        return horizontal_flip(img)
    if transformation == 1:
        return random_crop(img)
    if transformation == 2:
        return color_jitter(img)
    if transformation == 3:
        return random_resized_crop(img)
    if transformation == 4:
        return random_rotation(img)
    if transformation == 5:
        return random_affine_transformation(img)


def horizontal_flip(image):
    """Flip image horizontally."""
    transform = Compose([
        RandomHorizontalFlip(p=1.0),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,))
    ])

    transform_deepfashion = Compose([
        RandomHorizontalFlip(p=1.0),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])

    img = None
    if image.mode == 'L':
        img = transform(image)
    if image.mode == 'RGB':
        img = transform_deepfashion(image)
    return img


def random_crop(image):
    """Crop Image."""
    transform = Compose([
        RandomCrop((20, 20)),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,))
    ])

    transform_deepfashion = Compose([
        RandomCrop((20, 20)),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    img = None
    if image.mode == 'L':
        img = transform(image)
    if image.mode == 'RGB':
        img = transform_deepfashion(image)
    return img


def color_jitter(image):
    """Apply color jitter."""
    transform = Compose([
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.02),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,))
    ])

    transform_deepfashion = Compose([
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.02),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    img = None
    if image.mode == 'L':
        img = transform(image)
    if image.mode == 'RGB':
        img = transform_deepfashion(image)
    return img


def random_resized_crop(image):
    """Randomly resize and crop image."""
    transform = Compose([
        RandomResizedCrop(40, scale=(0.2, 1.0), ratio=(0.75, 1.333)),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,))
    ])

    transform_deepfashion = Compose([
        RandomResizedCrop(40, scale=(0.2, 1.0), ratio=(0.75, 1.333)),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    img = None
    if image.mode == 'L':
        img = transform(image)
    if image.mode == 'RGB':
        img = transform_deepfashion(image)
    return img


def random_rotation(image):
    """Randomly rotate image."""
    transform = Compose([
        RandomRotation(45),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,))
    ])

    transform_deepfashion = Compose([
        RandomRotation(45),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    img = None
    if image.mode == 'L':
        img = transform(image)
    if image.mode == 'RGB':
        img = transform_deepfashion(image)
    return img


def random_affine_transformation(image):
    """Applies a random affine transformation to the image."""
    transform = Compose([
        RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.3), shear=10),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,))
    ])

    transform_deepfashion = Compose([
        RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.3), shear=10),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    img = None
    if image.mode == 'L':
        img = transform(image)
    if image.mode == 'RGB':
        img = transform_deepfashion(image)
    return img


def train_exemplar_cnn():
    """Trains the exemplar cnn model."""
    print("=============================================================")
    print("============ Train ExemplarCNN with FashionMNIST ============")
    print("=============================================================\n")

    # number of predicted classes = number of training images
    exemplar_cnn = CifarNet(input_channels=1, num_classes=train_loader_fashion_mnist.batch_size)
    exemplar_cnn = exemplar_cnn.to(DEVICE)

    # Criteria NLLLoss which is recommended with softmax final layer
    loss_fn = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(exemplar_cnn.parameters(), lr=LEARNING_RATE_TRAIN)

    # Decay LR by a factor of 0.1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=STEP_SIZE_TRAIN, gamma=GAMMA)

    return train(exemplar_cnn, loss_fn, optimizer, scheduler, EPOCHS, train_loader_fashion_mnist)


def train_exemplar_cnn_deep_fashion():
    """Trains the exemplar cnn model."""
    print("============================================================")
    print("============ Train ExemplarCNN with DeepFashion ============")
    print("============================================================\n")

    # number of predicted classes = number of training images
    exemplar_cnn = CifarNet(input_channels=3, num_classes=train_loader_deep_fashion.batch_size)
    exemplar_cnn = exemplar_cnn.to(DEVICE)

    # Criteria NLLLoss which is recommended with softmax final layer
    loss_fn = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(exemplar_cnn.parameters(), lr=LEARNING_RATE_TRAIN)

    # Decay LR by a factor of 0.1 every 4 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=STEP_SIZE_TRAIN, gamma=GAMMA)

    return train(exemplar_cnn, loss_fn, optimizer, scheduler, EPOCHS, train_loader_deep_fashion)


def fine_tune_exemplar_cnn(model):
    """Fine tunes the exemplar cnn model."""
    print("=============================================================")
    print("========= Fine Tune Exemplar CNN with FashionMNIST ==========")
    print("=============================================================\n")

    # Criteria NLLLoss which is recommended with Softmax final layer
    loss_fn = nn.CrossEntropyLoss()

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc3.parameters():
        param.requires_grad = True

    # add fully connected layer and final layer with 10 outputs
    model.fc3 = nn.Sequential(nn.Linear(192, 192),
                              nn.Linear(192, 10, bias=True)
                              )

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_FINE_TUNE, weight_decay=WEIGHT_DECAY)

    # Decay LR by a factor of 0.1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=STEP_SIZE_FINE_TUNE, gamma=GAMMA)

    model = model.to(DEVICE)
    return fine_tune(model, loss_fn, optimizer, scheduler, EPOCHS_FINE_TUNE, train_loader_fashion_mnist, val_loader_fashion_mnist)


def fine_tune_exemplar_cnn_deep_fashion(model):
    """Fine tunes the exemplar cnn model."""
    print("============================================================")
    print("========= Fine Tune Exemplar CNN with DeepFashion ==========")
    print("============================================================\n")

    # Criteria NLLLoss which is recommended with Softmax final layer
    loss_fn = nn.CrossEntropyLoss()

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc3.parameters():
        param.requires_grad = True

    # add fully connected layer and final layer with 10 outputs
    model.fc3 = nn.Sequential(nn.Linear(192, 192),
                              nn.Linear(192, 50, bias=True)
                              )

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_FINE_TUNE)

    # Decay LR by a factor of 0.1 every 4 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=STEP_SIZE_FINE_TUNE, gamma=GAMMA)

    model = model.to(DEVICE)
    return fine_tune(model, loss_fn, optimizer, scheduler, EPOCHS_FINE_TUNE, train_loader_deep_fashion, val_loader_deep_fashion)


def test_classification_on_exemplar_cnn(model):
    """Fine tunes the exemplar cnn model."""
    print("=============================================================")
    print("=== Test Classification on Exemplar CNN with FashionMNIST ===")
    print("=============================================================\n")

    loss_fn = nn.CrossEntropyLoss()
    model = model.to(DEVICE)
    return test(model, loss_fn, test_loader_fashion_mnist)


def test_classification_on_exemplar_cnn_deep_fashion(model):
    """Fine tunes the exemplar cnn model."""
    print("============================================================")
    print("=== Test Classification on Exemplar CNN with DeepFashion ===")
    print("============================================================\n")

    loss_fn = nn.CrossEntropyLoss()
    model = model.to(DEVICE)
    return test(model, loss_fn, test_loader_deep_fashion)


def train(model, loss_fn, optimizer, scheduler, num_epochs, train_loader):
    """Train the model"""

    train_losses, train_accuracies = [], []
    best_acc = 0.0
    since = time.time()

    for epoch in range(num_epochs):
        scheduler.step()
        model.train()

        running_loss = []
        running_corrects = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            images_transformed, labes_transformed = [], []

            for index, img in enumerate(images):
                transformed_imgs = [
                    img,
                    transform_image(img, 0),
                    transform_image(img, 1),
                    transform_image(img, 2),
                    transform_image(img, 3),
                    transform_image(img, 4),
                    transform_image(img, 5),
                ]
                transformed_labels = torch.LongTensor([index, index, index, index, index, index, index])
                stack = torch.stack(transformed_imgs, dim=0)

                images_transformed.append(stack)
                labes_transformed.append(transformed_labels)

            images = torch.cat(images_transformed, dim=0).to(DEVICE)
            labels = torch.cat(labes_transformed, dim=0).to(DEVICE)

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
            running_corrects += torch.sum(preds == labels.data).to(torch.float32)

        train_losses.append(np.mean(np.array(running_loss)))
        train_accuracies.append((100.0 * running_corrects) / (7 * len(train_loader.dataset)))

        print('Epoch {}/{}: train_loss: {:.4f}, train_accuracy: {:.4f}'.format(
            epoch + 1, num_epochs,
            train_losses[-1],
            train_accuracies[-1]))

    time_elapsed = time.time() - since

    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best accuracy: {:4f}'.format(best_acc))

    return model, train_losses, train_accuracies
