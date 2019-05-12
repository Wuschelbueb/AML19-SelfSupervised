
import numpy as np


def rotate_imgGreyScale(img, rot):
    if rot == 0:
        rotatedImage = img
    if rot == 90:
        rotatedImage = np.fliplr(np.transpose(img))
    if rot == 180:
        rotatedImage = np.fliplr((np.flipud(img)))
    if rot == 270:
        rotatedImage = np.flipud(np.transpose(img))
    return rotatedImage


def rotate_img(img, rot):
    if rot == 0:
        return img
    elif rot == 90:
        return np.flipud(np.transpose(img, (1, 0, 2)))
    elif rot == 180:
        return np.fliplr(np.flipud(img))
    elif rot == 270:
        return np.transpose(np.flipud(img), (1, 0, 2))
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')



def createRotatedImg(pics):
    # todo: think of optimizing by extending pics and not copying it again
    # todo: do not rotate with an angle of 0 degrees
    rotPics = []
    rotLabels = []
    rot = [0, 90, 180, 270]

    # rotate all images in four directions
    rotInx = 0
    for rotVal in rot:
        for pic in pics:
            rotPics.append(rotate_img(pic, rotVal))
            rotLabels.append(rotVal)
    return rotPics, rotLabels