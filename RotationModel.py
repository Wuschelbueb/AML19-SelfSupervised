import torch
import torch.nn as nn
from train import train
from DataHandler import train_val_dataloader, train_data_rotation_loader, val_data_rotation_loader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def flatten(x):
    return x.view(x.size(0), -1)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.stride = stride

    def forward(self, x):

        residue = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            residue = self.downsample(x)

        out += residue
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.downsample = downsample

        self.stride = stride

    def forward(self, x):
        residue = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(x)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residue = self.downsample(x)

        out += residue
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, depth, name, num_classes=10, block=BasicBlock):
        super(ResNet, self).__init__()

        assert (depth - 2) % 6 == 0, 'Depth should be 6n + 2'
        n = (depth - 2) // 6

        self.name = name
        block = BasicBlock
        self.inplanes = 16
        fmaps = [16, 32, 64]  # CIFAR10

        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, fmaps[0], n, stride=1)
        self.layer2 = self._make_layer(block, fmaps[1], n, stride=2)
        self.layer3 = self._make_layer(block, fmaps[2], n, stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.flatten = flatten
        self.fc = nn.Linear(fmaps[2] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        ''' Between layers convolve input to match dimensions -> stride = 2 '''

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x, print_sizes=False):
        if print_sizes:
            print('Sizes of the tensors inside each node: \n')
            print("\t In Model: input size", x.size())

        x = self.relu(self.bn(self.conv(x)))  # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)  # 1x1
        x = self.flatten(x)  # Flatten
        x = self.fc(x)      # Dense

        if print_sizes:
            print("\t In Model: output size", x.size())

        return x


def ResNet20(**kwargs):
    return ResNet(name='ResNet20', depth=20, num_classes=4, **kwargs)


def train_rotation_model(model):
    # fitting the convolution to 1 input channel (instead of 3)
    model.conv = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    # Criteria NLLLoss which is recommended with Softmax final layer
    loss_fn = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 4 epochs
    sched = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=4, gamma=0.1)

    # Number of epochs
    eps = 10

    train_loader = train_data_rotation_loader()
    val_loader = val_data_rotation_loader()
    return train(model, loss_fn, optim, sched, eps, train_loader, val_loader)


def fine_tune_fc(model):
    # Criteria NLLLoss which is recommended with Softmax final layer
    loss_fn = nn.CrossEntropyLoss()

    # freeze all layers of the trained model
    for param in model.layer1.parameters():
        param.requires_grad = False

    for param in model.layer2.parameters():
        param.requires_grad = False

    for param in model.layer3.parameters():
        param.requires_grad = False

    # unfreeze final fc layer
    for param in model.fc.parameters():
        param.requires_grad = True

    # replace fc layer with 10 outputs
        model.fc = nn.Linear(64, 10)

    # Observe that all parameters are being optimized
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 4 epochs
    sched = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=4, gamma=0.1)

    # Number of epochs
    eps = 10

    model = model.to(device)
    train_loader, val_loader = train_val_dataloader()
    return train(model, loss_fn, optim, sched, eps, train_loader, val_loader)


def fine_tune_variant_2(model):
    loss_fn = nn.CrossEntropyLoss()

    # freeze all layers of the trained model
    for param in model.layer1.parameters():
        param.requires_grad = False

    for param in model.layer2.parameters():
        param.requires_grad = False

    for param in model.layer3.parameters():
        param.requires_grad = True

    # unfreeze final fc layer
    for param in model.fc.parameters():
        param.requires_grad = True

    # replace fc layer with 10 outputs
        model.fc = nn.Linear(64, 10)

    # Observe that all parameters are being optimized
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 4 epochs
    sched = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=4, gamma=0.1)

    # Number of epochs
    eps = 10

    model = model.to(device)
    train_loader, val_loader = train_val_dataloader()
    return train(model, loss_fn, optim, sched, eps, train_loader, val_loader)

