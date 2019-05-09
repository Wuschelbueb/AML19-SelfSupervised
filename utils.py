# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:49:27 2019

@author: Gwenael
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_n_curves(values, legend,title="n curves graph",  axis1 = "epochs", axis2= "loss/val"):
    """
    Plot n curves in the same graph. use as follows
    
    plot_n_curves([[1,2,3,4,5],[4,5,4,5,4],[2,2,2,2,2]], ["growing", "oscillating", "constant"], title = "3 different curves", axis1 = "epochs", axis2= "loss/val")    
    """
    plt.figure()
    for v in values:
         plt.plot(np.arange(len(v)), v)
    plt.legend(legend)
    plt.xlabel(axis1)
    plt.ylabel(axis2)
    plt.title(title)
    
    