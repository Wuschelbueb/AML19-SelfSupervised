import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


ROOT_DIR_FASHION_MNIST = 'fashion_mnist/'
ROOT_DIR_DEEP_FASHION = ''
PLOT_DIRECTORY = './Plots/'

EPOCHS = 25
EPOCHS_FINE_TUNE = 10

# Optimizer & Scheduler parameter

LEARNING_RATE_TRAIN = 0.001
LEARNING_RATE_FINE_TUNE = 0.0001

WEIGHT_DECAY = 1e-4
STEP_SIZE_TRAIN = EPOCHS//2
STEP_SIZE_FINE_TUNE = EPOCHS_FINE_TUNE//2
GAMMA = 0.1

TARGET_SIZE = (32, 32)
BATCH_SIZE = 256

TRANSFORM_DEEP_FASHION = Compose([Resize(TARGET_SIZE), ToTensor(), Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
TRANSFORM_FASHION_MNIST = Compose([Resize(TARGET_SIZE), ToTensor(), Normalize(mean=(0.5,), std=(0.5,))])

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
