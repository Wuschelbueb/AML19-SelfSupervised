"""Contains the code for the exemplar cnn sub task."""
from random import randint
import time
import torch
import torch.nn as nn
import numpy as np
from cifar_net import CifarNet
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, ColorJitter, \
    RandomResizedCrop, RandomRotation, RandomAffine, Compose, Resize, ToTensor, \
    Normalize, ToPILImage
from fashion_mnist_data_handler import train_loader_classification, val_loader_classification, \
    test_loader_classification, train_loader_exemplar_cnn, test_loader_exemplar_cnn
from train import train_and_val
from test import test

EPOCHS = 15
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader_classification = train_loader_classification()
val_loader_classification = val_loader_classification()
test_loader_classification = test_loader_classification()

train_loader_exemplar_cnn = train_loader_exemplar_cnn()
test_loader_exemplar_cnn = test_loader_exemplar_cnn()


def transform_images(images):
    """Transforms all images of the given batch."""
    for index, img in enumerate(images):
        images[index] = random_transform(img)
    return images


def random_transform(image):
    """Randomly transforms one image."""
    transform = ToPILImage()
    img = transform(image)

    transformation = randint(0, 5)

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
    """Flip image horiontally."""
    transform = Compose([
        RandomHorizontalFlip(p=1.0),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,))
    ])
    img = transform(image)
    return img


def random_crop(image):
    """Crop Image."""
    transform = Compose([
        RandomCrop((20, 20)),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,))
    ])
    img = transform(image)
    return img


def color_jitter(image):
    """Apply color jitter."""
    transform = Compose([
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.02),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,))
    ])
    img = transform(image)
    return img


def random_resized_crop(image):
    """Randomly resize and crop image."""
    transform = Compose([
        RandomResizedCrop(40, scale=(0.2, 1.0), ratio=(0.75, 1.333)),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,))
    ])
    img = transform(image)
    return img


def random_rotation(image):
    """Randomly rotate image."""
    transform = Compose([
        RandomRotation(45),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,))
    ])
    img = transform(image)
    return img


def random_affine_transformation(image):
    """Applies a random affine transformation to the image."""
    transform = Compose([
        RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.3), shear=10),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,))
    ])
    img = transform(image)
    return img


def train_exemplar_cnn():
    """Trains the exemplar cnn model."""
    print("===========================================")
    print("============ Train ExemplarCNN ============")
    print("===========================================\n")

    # exemplar_cnn = ResNet20ExemplarCNN()
    exemplar_cnn = CifarNet(input_channels=1, num_classes=4)
    exemplar_cnn = exemplar_cnn.to(device)

    # fitting the convolution to 1 input channel (instead of 3)
    exemplar_cnn.conv = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)

    # Criteria NLLLoss which is recommended with softmax final layer
    loss_fn = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(exemplar_cnn.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 4 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=4, gamma=0.1)

    return train(exemplar_cnn, loss_fn, optimizer, scheduler, EPOCHS, train_loader_exemplar_cnn)


def fine_tune_exemplar_cnn(model):
    """Fine tunes the exemplar cnn model."""
    print("===========================================")
    print("========= Fine Tune Exemplar CNN ==========")
    print("===========================================\n")

    # Criteria NLLLoss which is recommended with Softmax final layer
    loss_fn = nn.CrossEntropyLoss()

    # freezes all layers except the final one, according to the method parameters

    # for param in model.conv1.parameters():
    #     param.requires_grad = False
    #
    # for param in model.max1.parameters():
    #     param.requires_grad = False
    #
    # for param in model.batch1.parameters():
    #     param.requires_grad = False
    #
    # for param in model.conv2.parameters():
    #     param.requires_grad = False
    #
    # for param in model.batch2.parameters():
    #     param.requires_grad = False
    #
    # for param in model.max2.parameters():
    #     param.requires_grad = False
    #
    # for param in model.fc1.parameters():
    #     param.requires_grad = False
    #
    # for param in model.drop.parameters():
    #     param.requires_grad = False
    #
    # for param in model.fc2.parameters():
    #     param.requires_grad = False

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
    return train_and_val(model, loss_fn, optimizer, scheduler, EPOCHS, train_loader_classification,
                         val_loader_classification)


def test_classification_on_exemplar_cnn(model):
    """Fine tunes the exemplar cnn model."""
    print("===========================================")
    print("=== Test Classification on Exemplar CNN ===")
    print("===========================================\n")

    # Criteria NLLLoss which is recommended with Softmax final layer
    loss_fn = nn.CrossEntropyLoss()

    # replace fc layer with 10 outputs
    model.fc3 = nn.Linear(192, 10, bias=True)

    model = model.to(device)
    return test(model, loss_fn, EPOCHS, test_loader_classification)


def train(model, loss_fn, optimizer, scheduler, num_epochs, train_loader):
    """Train the model"""

    train_losses = []
    train_accuracies = []

    best_model_wts = model.state_dict()
    best_acc = 0.0

    since = time.time()

    for epoch in range(num_epochs):
        scheduler.step()
        model.train()

        running_loss = []
        running_corrects = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # apply random transformation to image, but don't change label
            transformed_images = transform_images(images)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(transformed_images)
            _, preds = torch.max(outputs.data, 1)
            loss = loss_fn(outputs, labels)

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

            # statistics
            running_loss.append(loss.item())
            running_corrects += torch.sum(preds == labels.data).to(torch.float32)

        train_losses.append(np.mean(np.array(running_loss)))
        train_accuracies.append(100.0 * running_corrects / len(train_loader.dataset))

        # deep copy the model
        if running_corrects > best_acc:
            best_acc = running_corrects
            best_model_wts = model.state_dict()

        print('Epoch {}/{}: train_loss: {:.4f}, train_accuracy: {:.4f}'.format(
            epoch + 1, num_epochs,
            train_losses[-1],
            train_accuracies[-1]))

    time_elapsed = time.time() - since
    model.load_state_dict(best_model_wts)  # load best model weights

    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model, train_losses, train_accuracies
