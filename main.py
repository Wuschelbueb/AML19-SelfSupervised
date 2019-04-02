
import imageRotatorService
import imageLoader
import imageRotatorService
import dataManipulator
import plotService
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from sklearn.model_selection import train_test_split
import dataContainer
from torch.utils.data import DataLoader

import torch.nn as nn
from torchvision import models
from torch import optim



# load all jpgs into a list of numpy arrays
folderWithJpgs = 'data/img'
pics = imageLoader.loadallJPGToNumpy(folderWithJpgs)
#plotService.plotNumpyPics(pics[1:4])


# create rotrated images and the labels
rotImg, rotLabels = imageRotatorService.createRotatedImg(pics)

# split in train and validation sets
X_train, X_val, y_train, y_val = train_test_split(rotImg, rotLabels, test_size=0.1, random_state=42)


# Prepare data / crop and normalize and move to pytorch tensor
imgTensorTrain = dataManipulator.preprocess(X_train)
imgTensorVal = dataManipulator.preprocess(X_val)
labelsTrain = torch.Tensor(y_train)
labelsVal = torch.Tensor(y_val)


# create data set objects for train and validation set
dataSetTrain = dataContainer.ImageDataset(imgTensorTrain, labelsTrain)
dataSetVal = dataContainer.ImageDataset(imgTensorVal, labelsVal)

# create data loader objects for train and validation sets
dataloader_train = DataLoader(dataSetTrain, batch_size=10, shuffle=True, num_workers=32)
dataloader_val = DataLoader(dataSetVal, batch_size=10, shuffle=True, num_workers=4)


# define model
model = models.resnet50(pretrained=True)

# freezing the weights of the pretrained model
for param in model.parameters():
    param.requires_grad = False

# add layers on top of trained resnet
# two linear layers with a final output of 4 classes and a softmax for probability calc.
model.fc = nn.Sequential(nn.Linear(2048,512),
                         nn.ReLU(),
                         nn.Dropout(0.2),
                         nn.Linear(512, 4),
                         nn.LogSoftmax(dim=1))

# choose loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)



# create proper device, cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# move to device
model.to(device)


# train model

epoches = 2
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []
model.train()
for epoch in range(epoches):
    print('epoch:', epoch)
    for inputs, labels in dataloader_train:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        print('loss =', loss)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()



        # # after certain steps print loss
        # if steps % print_every == 0:
        #     test_loss = 0
        #     accuracy = 0
        #     model.eval()
        #     with torch.no_grad():
        #         for inputsVal, labelsVal in dataloader_val:
        #             inputsVal, labelsVal = inputs.to(device), labels.to(device)
        #             logps = model.forward(inputsVal)
        #             batch_loss = criterion(logps, labelsVal)
        #             test_loss += batch_loss.item()
        #             ps = torch.exp(logps)
        #             top_p, top_class = ps.topk(1, dim=1)
        #             equals = top_class == labelsVal.view(*top_class.shape)
        #             accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        #     train_losses.append(running_loss / len(dataloader_train))
        #     test_losses.append(test_loss / len(dataloader_val))
        #     print('Epoch:', (epoch+1)/epoches)
        #     print('Train loss:', running_loss/print_every)
        #     print('Val loss:', test_loss / len(dataloader_val))
        #     print('Val accuracy:', accuracy / len(dataloader_val))
        #     running_loss = 0

