"""Fine tune methods."""
import time

import numpy as np
import torch

from settings import DEVICE


def fine_tune(model, loss_fn, optimizer, scheduler, num_epochs, train_loader, val_loader):
    """Fine tune the model"""
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
        train_accuracies.append(100.0 * running_corrects_train / len(train_loader.dataset))

        model.eval()
        running_corrects_val = 0.0
        running_loss = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                # forward
                outputs = model(images)
                _, preds = torch.max(outputs.data, 1)
                loss = loss_fn(outputs, labels)

                # statistics
                running_loss.append(loss.item())
                running_corrects_val += torch.sum(preds == labels.data).to(torch.float32)

        val_losses.append(np.mean(np.array(running_loss)))
        val_accuracies.append(100.0 * running_corrects_val / len(val_loader.dataset))

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
