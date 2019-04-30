import time
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, loss_fn, optimizer, scheduler, num_epochs, train_loader, val_loader):
    """Train the model"""

    best_model_wts = model.state_dict()
    best_acc = 0.0
    data_loader = None
    dataset_size = 0

    for epoch in range(num_epochs):
        since = time.time()

        print('Epoch {}/{}'.format(epoch + 1, num_epochs))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
                data_loader = train_loader
                dataset_size = len(train_loader.dataset)
            else:
                model.train(False)  # Set model to evaluate mode
                data_loader = val_loader
                dataset_size = len(val_loader.dataset)

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.

            for data in data_loader:
                # get the inputs
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = loss_fn(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).to(torch.float32)

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects / dataset_size

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
