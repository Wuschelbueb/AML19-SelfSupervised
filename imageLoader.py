import os
from PIL import Image
import numpy as np



def loadallJPGToNumpy(parentFolder):
    X = []
    # loop through all subdirs and get all jpgs and load them into features matrix
    for dirpath, dirnames, filenames in os.walk(parentFolder):
        for filename in [f for f in filenames if f.endswith(".jpg")]:
            # convert jpg to numpy array
            source = dirpath + "/" + filename
            #print(source)
            X.append(np.array(Image.open(source)))

    return X