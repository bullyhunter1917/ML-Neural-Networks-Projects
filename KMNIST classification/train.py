import argparse
import time
import torch
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from torch import nn
from torch.optim import Adam
from torch.utils.data import random_split
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.utils.data import DataLoader
from lenet import lenet

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to output trained model")
ap.add_argument("-p", "--plot", type=str, required=True,
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())


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

for i in tqdm(range(0, EPOCHS)):
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
        trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

    # Testing Validation set
    with torch.no_grad():
        # Setting model into evaluation mode
        model.eval()

        for (x, y) in _valDataLoader:
            (x, y) = (x.to(device), y.to(device))

            pred = model(x)
            ValLoss += lossFunction(pred, y)

            valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

    # Calculating and saving data for loss/acc plot
    avgTrainLoss = TrainLoss / _trainStep
    avgValLoss = ValLoss / _valStep

    trainCorrect = trainCorrect / len(_trainDataLoader.dataset)
    valCorrect = valCorrect / len(_valDataLoader.dataset)

    History["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    History["train_acc"].append(trainCorrect)
    History["val_loss"].append(avgValLoss.cpu().detach().numpy())
    History["val_acc"].append(valCorrect)

end_time = time.time()

print(f'Time taken to train mode {(end_time-start_time)}')

# Evaluating test set
with torch.no_grad():
    # changing model into eval mode
    model.eval()

    pred_res = []

    # predicting for test set
    for (x, y) in _testDataLoader:
        x = x.to(device)

        pred = model(x)
        pred_res.extend(pred.argmax(axis=1).cpu().numpy())


# Printing classification report
report = classification_report(_test.targets.cpu().numpy(), np.array(pred_res), target_names=_test.classes)
print(report)

# Creating loss and acc plot
plt.figure()
plt.plot(History["train_loss"], label="train_loss")
plt.plot(History["val_loss"], label="val_loss")
plt.plot(History["train_acc"], label="train_acc")
plt.plot(History["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on KMNIST")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# saving model to disc
torch.save(model, args["model"])