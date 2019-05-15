"""Creates the data set."""
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

TARGET_SIZE = (32, 32)
TRANSFORM = Compose([ Resize(TARGET_SIZE), ToTensor(), Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])


# class DeepFashionDataset(Dataset):
#     """Creates the DeepFashion data set."""
#     def __init__(self, data, target):
#         self.data = data
#         self.target = target
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         data = self.data[index][0]
#         target = self.target[index]
#
#         return data, target


def default_loader(path):
    return Image.open(path)


class DeepFashionDataset(Dataset):
    def __init__(self, root, flist, targets, transform=TRANSFORM, loader=default_loader):
        self.root = root
        self.imlist = flist
        self.targets = targets
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        try:
            impath = self.imlist[index]
            img = self.loader(os.path.join(self.root, impath))
            target = self.targets[index]
            if self.transform is not None:
                img = self.transform(img)
            return img, target
        except IndexError:
            print('do nothing')

    def __len__(self):
        return len(self.imlist)

