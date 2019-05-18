import argparse

import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate_train', default=0.001, type=float)
parser.add_argument('--learning_rate_fine_tune', default=0.001, type=float)
parser.add_argument('--epochs_train', default=15, type=int)
parser.add_argument('--epochs_fine_tune', default=5, type=int)
args = parser.parse_args()


ROOT_DIR_FASHION_MNIST = 'fashion_mnist/'
ROOT_DIR_DEEP_FASHION = ''
PLOT_DIRECTORY = './Plots/'

EPOCHS = args.epochs_train
EPOCHS_FINE_TUNE = args.epochs_fine_tune

# Optimizer & Scheduler parameter
LEARNING_RATE_TRAIN = args.learning_rate_train
LEARNING_RATE_FINE_TUNE = args.learning_rate_fine_tune

SIZE_DEEP_FASHION_TRAIN = 60000
SIZE_DEEP_FASHION_VAL = 20000
SIZE_DEEP_FASHION_TEST = 20000

WEIGHT_DECAY = 1e-4
STEP_SIZE_TRAIN = EPOCHS//2
STEP_SIZE_FINE_TUNE = EPOCHS_FINE_TUNE//2
GAMMA = 0.1

TARGET_SIZE = (32, 32)
BATCH_SIZE = 256

TRANSFORM_DEEP_FASHION = Compose([Resize(TARGET_SIZE), ToTensor(), Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
TRANSFORM_FASHION_MNIST = Compose([Resize(TARGET_SIZE), ToTensor(), Normalize(mean=(0.5,), std=(0.5,))])

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
