from __future__ import print_function
from PIL import Image, PILLOW_VERSION
import os
import os.path
import numpy as np
import sys
import numbers
import random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
from torchvision.transforms.functional import _get_inverse_affine_matrix
import math

class FashionMNIST(data.Dataset):
    def __init__(self, root, scale=None, download=False, train=True, fillcolor=0,
            transform_pre=None, matrix_transform=None, transform=None, target_transform=None,
            resample=False):
        self.root = os.path.expanduser(root)
        self.transform_pre = transform_pre
        self.transform = transform
        self.target_transform = target_transform
        self.matrix_transform = matrix_transform
        self.train = train #can be used for training or testing

        if download:
            self.download()

        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC


            
        return super().__init__(*args, **kwargs)