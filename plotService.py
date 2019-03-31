import matplotlib
import matplotlib.pyplot as plt

def plotNumpyPics(pics):
    for pic in pics:
        plt.figure()
        plt.imshow(pic)