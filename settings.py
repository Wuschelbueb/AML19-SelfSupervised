import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


ROOT_DIR_FASHION_MNIST = 'fashion_mnist/'
ROOT_DIR_DEEP_FASHION = ''
PLOT_DIRECTORY = './Plots/'

EPOCHS = 1
TARGET_SIZE = (32, 32)
BATCH_SIZE = 64

TRANSFORM_DEEP_FASHION = Compose([Resize(TARGET_SIZE), ToTensor(), Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
TRANSFORM_FASHION_MNIST = Compose([Resize(TARGET_SIZE), ToTensor(), Normalize(mean=(0.5,), std=(0.5,))])

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
