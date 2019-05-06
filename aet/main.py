import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as TF
import time
import PIL
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.autograd import Variable
from torchvision.models import resnet18
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Normalize, RandomRotation, ToPILImage
from matplotlib import pyplot as plt
from random import randint
from matplotlib import pyplot as plt
from random import randint
from NetworkInNetwork import Regressor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FashionMNISTDataset(Dataset):
    
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        data = self.data[index]
        target = self.targets[index]

        return data, target

def run():
    root_dir = '../fashion_mnist'
    loss = nn.MSELoss()

    # Download and load the training data
    trainset = FashionMNIST(
        root=root_dir, 
        scale=(0.8, 1.2),
        download=True, 
        train=True,
        fillcolor=(128,128,128), # ?
        resample=PIL.Image.BILINEAR,
        matrix_transform=transforms.Compose([
            transforms.Normalize((0., 0., 16., 0., 0., 16., 0., 0.), (1., 1., 20., 1., 1., 20., 0.015, 0.015)),
        ]),
        transform_pre=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]),
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]))

    # create datasets
    nbr_train_examples = round(len(trainset) * 0.8)
    train_dataset = FashionMNISTDataset(
        data=trainset.data[:nbr_train_examples], 
        targets=trainset.targets[:nbr_train_examples]
    )
    val_dataset = FashionMNISTDataset(
        data=trainset.data[nbr_train_examples:], 
        targets=trainset.targets[:nbr_train_examples:]
    )

    # create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=1)

    # Download and load the test data
    test_dataset = FashionMNIST(root=root_dir, download=True, train=False, transform=Compose([
        ToTensor(), Normalize(mean=(0.5,), std=(0.5,))]))
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=1)

    # resnet20 = ResNet20()
    # fitting the convolution to 1 input channel (instead of 3)
    # resnet20.conv = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    # Criteria NLLLoss which is recommended with Softmax final layer
    loss_fn = nn.CrossEntropyLoss()

    #import from NetworkInNetwork
    net = Regressor(_num_stages=4, _use_avg_on_conv3=False).to(device)
    if (device == 'cuda'):
        net = torch.nn.DataParallel(net, device_ids=range(1))
    
    net.load_state_dict(torch.load(''))

    print(net)

    # Observe that all parameters are being optimized
    # setup optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)

    # Decay LR by a factor of 0.1 every 4 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=4, gamma=0.1)

    # Number of epochs
    eps=1

    resnet20 = resnet20.to(device)
    resnet20_trained = train(resnet20, loss_fn, optimizer, scheduler, eps, train_loader, val_loader)

if __name__ == '__main__':
    run()