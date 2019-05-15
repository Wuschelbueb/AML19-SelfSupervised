"""Creates the data set."""
import os

from PIL import Image
from torch.utils.data import Dataset

from settings import TRANSFORM_DEEP_FASHION


def default_loader(path):
    """Data loader."""
    return Image.open(path)


class DeepFashionDataset(Dataset):
    """Creates the DeepFashion data set."""
    def __init__(self, root, image_list, targets, transform=TRANSFORM_DEEP_FASHION, loader=default_loader):
        self.root = root
        self.image_list = image_list
        self.targets = targets
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        try:
            image_path = self.image_list[index]
            img = self.loader(os.path.join(self.root, image_path))
            target = self.targets[index]
            if self.transform is not None:
                img = self.transform(img)
            return img, target
        except IndexError:
            print('do nothing')

    def __len__(self):
        return len(self.image_list)
