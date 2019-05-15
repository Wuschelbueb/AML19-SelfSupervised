"""Test methods."""
import torch

from settings import DEVICE


def test(model, loss_fn, test_loader):
    """Tests the model on data from test_loader"""
    model.eval()
    test_loss = 0
    n_correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            output = model(images)
            loss = loss_fn(output, labels)
            test_loss += loss.item()
            n_correct += torch.sum(output.argmax(1) == labels).item()

    average_test_loss = test_loss / len(test_loader.dataset)
    average_test_accuracy = 100.0 * n_correct / len(test_loader.dataset)

    print('Test average loss: {:.4f}, accuracy: {:.3f}'.format(average_test_loss), average_test_accuracy)
    return average_test_loss, average_test_accuracy
