
import imageRotatorService
import imageLoader
import imageRotatorService
import plotService
import matplotlib.pyplot as plt


# load all jpgs into a list of numpy arrays
folderWithJpgs = 'data/img'
pics = imageLoader.loadallJPGToNumpy(folderWithJpgs)
#plotService.plotNumpyPics(pics[1:4])


# create rotrated images and the labels
rotImg, rotLabels = imageRotatorService.createRotatedImg(pics)


# into pytorch tensor framework

