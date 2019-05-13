"""Create data set."""
from torch.utils.data import Dataset
from PIL import Image, PILLOW_VERSION
import torchvision.transforms as transforms
import numpy as np
import PIL
import random
import numbers
import torch


class FashionMNISTAETDataset(Dataset):
    """Create FashionMNIST AET data set."""

    def __init__(self, data, fillcolor=0):
        self.data = data
        self.shift = 4
        self.resample = PIL.Image.BILINEAR
        self.scale = (0.8, 1.2)
        self.target_transform = None
        self.matrix_transform = transforms.Compose([
            transforms.Normalize((0., 0., 16., 0., 0., 16., 0., 0.), (1., 1., 20., 1., 1., 20., 0.015, 0.015)),
            ])
        self.transform_pre = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            ])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])
        self.fillcolor = (128)


    @staticmethod   
    def find_coeffs(pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
    
        A = np.matrix(matrix, dtype=np.float)
        B = np.array(pb).reshape(8)
    
        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)

    def __getitem__(self, index):
        img1 = self.data[index][0]
        target = self.data[index][1]

        width, height = img1.size
        center = (img1.size[0] * 0.5 + 0.5, img1.size[1] * 0.5 + 0.5)
        shift = [float(random.randint(-int(self.shift), int(self.shift))) for ii in range(8)]
        scale = random.uniform(self.scale[0], self.scale[1])
        rotation = random.randint(0,3)

        pts = [((0-center[0])*scale+center[0], (0-center[1])*scale+center[1]),
            ((width-center[0])*scale+center[0], (0-center[1])*scale+center[1]),
            ((width-center[0])*scale+center[0], (height-center[1])*scale+center[1]),
            ((0-center[0])*scale+center[0], (height-center[1])*scale+center[1])] 
        pts = [pts[(ii+rotation)%4] for ii in range(4)]
        pts = [(pts[ii][0]+shift[2*ii], pts[ii][1]+shift[2*ii+1]) for ii in range(4)]

        coeffs = self.find_coeffs(
            pts,
            [(0, 0), (width, 0), (width, height), (0, height)]
        )
        
        kwargs = {"fillcolor": self.fillcolor} if PILLOW_VERSION[0] == '5' else {}
        img2 = img1.transform((width, height), Image.PERSPECTIVE, coeffs, self.resample, **kwargs)

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        if self.target_transform is not None:
            target = self.target_transform(target)

        coeffs = torch.from_numpy(np.array(coeffs, np.float32, copy=False)).view(8, 1, 1)

        # label of the image (later used to compare this and the perdicted one)
        coeffs = self.matrix_transform(coeffs)

        return img1, img2, coeffs
        
    def __len__(self):
        return len(self.data)