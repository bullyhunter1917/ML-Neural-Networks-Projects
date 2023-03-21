from torch.utils.data import random_split
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
import torch


# initing training parameters
INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 10

TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading KMIST
_train = KMNIST(root='data', train=True, download=True, transform=ToTensor())
_test = KMNIST(root='data', train=False, download=True, transform=ToTensor())

# Splitting train set
trainSampleSize = int(len(_train) * TRAIN_SPLIT)
valSampleSize = int(len(_train) * VAL_SPLIT)
(_train, _val) = random_split(_train, [trainSampleSize, valSampleSize], generator=torch.Generator().manual_seed(42))

