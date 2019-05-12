"""Create data set."""
from torch.utils.data import Dataset


class FashionMNISTExemplarCNNDataset(Dataset):
    """Create FashionMNIST data set for the exemplar cnn task."""

    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index][0]
        target = index

        return data, target
