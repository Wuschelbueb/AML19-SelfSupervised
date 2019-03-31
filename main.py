
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


# load all jpgs into a list of numpy arrays
folderWithJpgs = 'data/img'
pics = imageLoader.loadallJPGToNumpy(folderWithJpgs)
#plotService.plotNumpyPics(pics[1:4])


# create rotrated images and the labels
rotImg, rotLabels = imageRotatorService.createRotatedImg(pics)


# Prepare data / crop, move to pytorch tensor and normalise
imgTensor = dataManipulator.preprocess(rotImg)


#