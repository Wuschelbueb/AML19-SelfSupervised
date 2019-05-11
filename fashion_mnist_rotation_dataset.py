"""Create data set."""
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf


class FashionMNISTRotationDataset(Dataset):
    """Create FashionMNIST data set for the rotation task."""

    def __init__(self, data, target, angle):
        self.data = data
        self.target = target
        self.angle = angle

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = rotate(self.data[index][0], self.angle)
        target = None

        if self.angle == 0:
            target = 0
        if self.angle == 90:
            target = 1
        if self.angle == 180:
            target = 2
        if self.angle == 270:
            target = 3

        return data, target


def rotate(image, angle):
    """Rotate the image by the specified angle"""
    image = tf.to_pil_image(image)
    image = tf.rotate(image, angle)
    image = tf.to_tensor(image)
    image = tf.normalize(image, (0.5,), (0.5,))
    return image
