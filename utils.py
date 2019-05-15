"""
Created on Tue May  7 14:49:27 2019

@author: Gwenael
"""
import os

import matplotlib
import numpy as np

from settings import PLOT_DIRECTORY

matplotlib.use('agg')
import matplotlib.pyplot as plt


def get_name(title):
    split = title.split(' ')
    s = '_'
    concat_title = s.join(split)
    return concat_title + '.pdf'


def plot_n_curves(values, legend, title="n curves graph", axis1="epochs", axis2="loss/val"):
    """
    Plot n curves in the same graph. use as follows
    
    plot_n_curves(
        [[1,2,3,4,5],[4,5,4,5,4],[2,2,2,2,2]],
        ["growing", "oscillating", "constant"],
        title = "3 different curves",
        axis1 = "epochs",
        axis2= "loss/val"
    )
    """

    fig = plt.figure()

    if isinstance(values[0], list):
        # if it is a list of lists
        for v in values:
            plt.plot(np.arange(len(v)), v)
    else:
        # else it is only one single list
        plt.plot(np.arange(len(values)), values)
    plt.legend(legend)
    plt.xlabel(axis1)
    plt.ylabel(axis2)
    plt.title(title)

    final_title = get_name(title)

    # Pick one of the following lines to uncomment
    # save_file = None
    save_file = os.path.join(PLOT_DIRECTORY, final_title)

    if save_file:
        plt.savefig(save_file)
        plt.close(fig)
    else:
        plt.show()


def plot_summary(values, const, legend, title="n curves graph", axis1="epochs", axis2="loss/val"):
    """
    Plot n curves in the same graph. use as follows

    plot_n_curves(
        [[1,2,3,4,5],[4,5,4,5,4],[2,2,2,2,2]],
        ["growing", "oscillating", "constant"],
        title = "3 different curves",
        axis1 = "epochs",
        axis2= "loss/val"
    )
    """

    fig = plt.figure()

    if isinstance(values[0], list):
        # if it is a list of lists
        for v in values:
            plt.plot(np.arange(len(v)), v)
    else:
        # else it is only one single list
        plt.plot(np.arange(len(values)), values)
    plt.axhline(y=const, color='r', linestyle='--')
    plt.legend(legend)
    plt.xlabel(axis1)
    plt.ylabel(axis2)
    plt.title(title)

    final_title = get_name(title)

    # Pick one of the following lines to uncomment
    # save_file = None
    save_file = os.path.join(PLOT_DIRECTORY, final_title)

    if save_file:
        plt.savefig(save_file)
        plt.close(fig)
    else:
        plt.show()
