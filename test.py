"""Test methods."""
import numpy as np
import torch

from settings import DEVICE


# def test(model, loss_fn, num_epochs, test_loader):
#     model.eval()
#
#     test_accuracies = []
#     test_losses = []
#
#     for epoch in range(num_epochs):
#         corrects = 0
#         losses = []
#
#         with torch.no_grad():
#             for images, labels in test_loader:
#
#                 # get the inputs
#                 images = images.to(device)
#                 labels = labels.to(device)
#
#                 # forward
#                 outputs = model(images)
#                 _, preds = torch.max(outputs.data, 1)
#                 print('preds: {}'.format(preds))
#                 loss = loss_fn(outputs, labels)
#                 print('loss: {}'.format(loss))
#
#                 # statistics
#                 losses.append(loss.item())
#                 correct = torch.sum(preds == labels.data).to(torch.float32)
#                 corrects += correct
#                 print('losses: {}'.format(losses))
#                 print('correct: {}'.format(correct))
#
#         test_losses.append(np.mean(losses))
#         test_accuracies.append(100.0 * corrects / len(test_loader.dataset))
#
#         print('Epoch {}/{}: test_loss: {:.4f}, test_accuracy: {:.4f}'.format(
#             epoch + 1, num_epochs, test_losses[-1], test_accuracies[-1]))
#
#     print('Test average loss: {:.4f}, accuracy: {:.3f}'.format(np.mean(test_losses), np.mean(test_accuracies)))
#     return test_losses, test_accuracies


def test(model, loss_fn, num_epochs, test_loader):
    """Tests the model on data from test_loader"""
    model.eval()
    test_losses, test_accuracies = [], []

    for epoch in range(num_epochs):
        test_loss = 0
        n_correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                output = model(images)
                loss = loss_fn(output, labels)
                test_loss += loss.item()
                print('loss: {}'.format(loss.item()))
                correct = torch.sum(output.argmax(1) == labels).item()
                print('correct: {}'.format(correct))
                n_correct += torch.sum(output.argmax(1) == labels).item()

        average_loss = test_loss / len(test_loader)
        accuracy = 100.0 * n_correct / len(test_loader.dataset)

        test_losses.append(test_loss)
        test_accuracies.append(accuracy)

        print('Epoch {}/{}: Test average loss: {:.4f}, accuracy: {:.3f}'.format(
            epoch + 1, num_epochs, average_loss, accuracy))

    print('Test average loss: {:.4f}, accuracy: {:.3f}'.format(np.mean(test_losses), np.mean(test_accuracies)))
    return test_losses, test_accuracies
