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
        self.conv1 = nn.ConvTranspose2d(input_channels, out_channels=64, kernel_size=(5, 5), stride=2, padding=2,
                                        output_padding=1)
        self.batch1 = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.ConvTranspose2d(input_channels, out_channels=64, kernel_size=(5, 5), stride=1, padding=0,
                                        output_padding=0)
        self.conv3 = nn.ConvTranspose2d(input_channels, out_channels=64, kernel_size=(5, 5), stride=2, padding=2,
                                        output_padding=1)
        self.conv4 = nn.ConvTranspose2d(input_channels, out_channels=out_channels, kernel_size=(5, 5), stride=1,
                                        padding=0, output_padding=0)
        self.sigmoid = nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = out.view(-1, 64, 5, 5)
        out = self.conv1(out)
        out = self.batch1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.sigmoid(out)
        return out
