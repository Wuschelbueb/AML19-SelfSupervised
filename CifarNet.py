import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def flatten(x):
    return x.view(x.size(0), -1)


class CifarNet(nn.Module):

    def __init__(self, inputChannels, numClasses=10):
        super(CifarNet, self).__init__()
        self.conv1  = nn.Conv2d(inputChannels, 64, kernel_size=(5, 5), bias=False)
        self.max1   = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.batch1 = nn.BatchNorm2d(num_features=64)
        self.conv2   = nn.Conv2d(64, 64, kernel_size=(5, 5), bias=False)
        self.batch2 = nn.BatchNorm2d(num_features=64)
        self.max2   = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.flat   = flatten
        self.fc1    = nn.Linear(1600, 384, bias=True)
        self.drop   = nn.Dropout(0.5)
        self.fc2    = nn.Linear(384, 192, bias=True)
        self.fc3    = nn.Linear(192, numClasses, bias=True)

    def forward(self, input):
        x = self.conv1(input)
        x = self.max1(x)
        x = self.batch1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.max2(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

