import torch.nn as nn


def flatten(x):
    """Flattens a tensor."""
    return x.view(x.size(0), -1)


class Decoder(nn.Module):
    """Decoder to CifarNet model"""

    def __init__(self, input_channels, num_classes=10, out_channels=1):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(num_classes, 192, bias=True)
        self.fc2 = nn.Linear(192, 384, bias=True)
        self.fc3 = nn.Linear(384, 1600, bias=True)
        self.conv1 = nn.ConvTranspose2d(input_channels, out_channels=64, kernel_size=(5, 5), stride=2, padding=2, output_padding=1)
        self.batch1 = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.ConvTranspose2d(input_channels, out_channels=64, kernel_size=(5, 5), stride=1, padding=0, output_padding=0)
        self.conv3 = nn.ConvTranspose2d(input_channels, out_channels=64, kernel_size=(5, 5), stride=2, padding=2, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(input_channels, out_channels=out_channels, kernel_size=(5, 5), stride=1, padding=0, output_padding=0)
        self.sigmoid = nn.Tanh()

    def forward(self, input):
        x = self.fc1(input)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.view(-1, 64, 5, 5)
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.sigmoid(x)
        return x