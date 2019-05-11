"""Create data set."""
from torch.utils.data import Dataset


class FashionMNISTDataset(Dataset):
    """Create FashionMNIST data set."""

    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index][0]
        target = self.targets[index][1]

        return data, target
