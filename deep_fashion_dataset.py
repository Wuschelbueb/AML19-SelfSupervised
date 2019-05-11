"""Creates the data set."""

from torch.utils.data import Dataset


class DeepFashionDataset(Dataset):
    """Creates the DeepFashion data set."""
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index][0]
        target = self.target[index]

        return data, target
