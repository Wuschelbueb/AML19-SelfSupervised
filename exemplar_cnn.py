"""Contains the code for the exemplar cnn sub task."""
from random import randint
import time
import torch
import torch.nn as nn
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, ColorJitter, \
    RandomResizedCrop, RandomRotation, RandomAffine, Compose, Resize, ToTensor, \
    Normalize, ToPILImage
from fashion_mnist_data_handler import train_loader_classification, val_loader_classification, \
    test_loader_classification, train_loader_exemplar_cnn, test_loader_exemplar_cnn
from resnet import ResNet20ExemplarCNN
from train import train_and_val

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

    exemplar_cnn = ResNet20ExemplarCNN()
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


def fine_tune_exemplar_cnn(model, unfreeze_l1, unfreeze_l2, unfreeze_l3, unfreeze_fc):
    """Fine tunes the exemplar cnn model."""
    print("===========================================")
    print("========= Fine Tune Exemplar CNN ==========")
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
    return train_and_val(model, loss_fn, optimizer, scheduler, EPOCHS, train_loader_classification,
                         val_loader_classification)


def train(model, loss_fn, optimizer, scheduler, num_epochs, train_loader):
    """Train the model"""
    for epoch in range(num_epochs):
        since = time.time()

        print('Epoch {}/{}\n'.format(epoch + 1, num_epochs))

        scheduler.step()
        model.train(True)  # Set model to training mode
        dataset_size = len(train_loader.dataset)

        running_loss = 0.0
        running_corrects = 0.0

        # Iterate over data.

        for data in train_loader:
            # get the inputs
            images, labels = data
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
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).to(torch.float32)

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects / dataset_size

        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    time_elapsed = time.time() - since

    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model