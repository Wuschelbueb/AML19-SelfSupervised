
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
dataloader_train = DataLoader(dataSetTrain, batch_size=100, shuffle=True, num_workers=32)
dataloader_val = DataLoader(dataSetVal, batch_size=100, shuffle=True, num_workers=4)
