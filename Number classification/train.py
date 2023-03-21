import time

from torch import nn
from torch.optim import Adam
from torch.utils.data import random_split
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.utils.data import DataLoader
import torch
from lenet import lenet

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

# Creating dataloaders for train val and test datasets
_trainDataLoader = DataLoader(_train, batch_size=BATCH_SIZE, shuffle=True)
_valDataLoader = DataLoader(_val, batch_size=BATCH_SIZE)
_testDataLoader = DataLoader(_test, batch_size=BATCH_SIZE)

_trainStep = len(_train) // BATCH_SIZE
_valStep = len(_val) // BATCH_SIZE

model = lenet.LeNet(numOfChannels=1, classes=len(_train.dataset.classes)).to(device)

optimizer = Adam(model.parameters(), lr=INIT_LR)
lossFunction = nn.NLLLoss()

# Creating history to make plots
History = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

# Measuring how long training will take
start_time = time.time()

for i in range(0, EPOCHS):
    # Setting model into train mode
    model.train()

    # Statistics for plot
    TrainLoss = 0
    ValLoss = 0
    trainCorrect = 0
    valCorrect = 0

    # Training model
    for (x, y) in _trainDataLoader:
        # Sending input to device
        (x, y) = (x.to(device), y.to(device))

        # Predict and calculate loss
        pred = model(x)
        loss = lossFunction(pred, y)

        # Change grad to zero from last epoch, calculate new grad and update values
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        TrainLoss += loss
        trainCorrect += (pred.argmax(1) == 1).type(torch.float).sum().item()

    # Testing Validation set
    with torch.no_grad():
        # Setting model into evaluation mode
        model.eval()

        for (x, y) in _valDataLoader:
            (x, y) = (x.to(device), y.to(device))

            pred = model(x)
            ValLoss += lossFunction(pred, y)

            valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()


