import torch
from torch.utils.data import Dataset



class ImageDataset(Dataset):

    # write your code
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        datapoint = self.data[index]
        label = self.labels[index]
        return datapoint, label

    def __len__(self):
       return len(self.data)