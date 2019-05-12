"""Test methods."""
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(model, loss_fn, num_epochs, test_loader):
    model.eval()

    test_accuracies = []
    test_losses = []

    for epoch in range(num_epochs):
        corrects = 0
        losses = []

        with torch.no_grad():
            for images, labels in test_loader:

                # get the inputs
                images = images.to(device)
                labels = labels.to(device)

                # forward
                outputs = model(images)
                _, preds = torch.max(outputs.data, 1)
                loss = loss_fn(outputs, labels)

                # statistics
                losses.append(loss.item())
                corrects += torch.sum(preds == labels.data).to(torch.float32)

        test_losses.append(np.mean(losses))
        test_accuracies.append(100.0 * corrects / len(test_loader.dataset))

        print('Epoch {}/{}: test_loss: {:.4f}, test_accuracy: {:.4f}'.format(
            epoch + 1, num_epochs, test_losses[-1], test_accuracies[-1]))

    print('Test average loss: {:.4f}, accuracy: {:.3f}'.format(np.mean(test_losses), np.mean(test_accuracies)))
    return test_losses, test_accuracies
