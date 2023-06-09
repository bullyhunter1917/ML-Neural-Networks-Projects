import argparse
import time
import numpy as np
import torch
import torchvision.transforms.functional
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from tqdm import tqdm
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from cnn import cnn
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', type=str, required=True, help='path to output trined model')
ap.add_argument('-p', '--plot', type=str, required=True, help='path to output loss/accuracy plot')
args = vars(ap.parse_args())

INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 10

TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading Food-101
_train = Food101(root='data', split='train', download=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor()
]))
_test = Food101(root='data', split='test',download=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor()
]))

# Splitting train set
trainSampleSize = int(len(_train) * TRAIN_SPLIT)
valSampleSize = int(len(_train) * VAL_SPLIT)

(_train, _val) = random_split(_train, [trainSampleSize, valSampleSize], generator=torch.Generator().manual_seed(42))

_trainDataLoader = DataLoader(_train, batch_size=BATCH_SIZE, shuffle=True)
_valDataLoader = DataLoader(_val, batch_size=BATCH_SIZE)
_testDataLoader = DataLoader(_test, batch_size=BATCH_SIZE)

_trainStep = len(_train) // BATCH_SIZE
_valStep = len(_val) // BATCH_SIZE


model = cnn.Cnn(numOfChannels=3, classes=len(_train.dataset.classes)).to(device)

optimizer = Adam(model.parameters(), lr=INIT_LR)
lossFunction = nn.NLLLoss()

History = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

start_time = time.time()

for i in tqdm(range(EPOCHS)):
    model.train()

    TrainLoss = 0
    ValLoss = 0
    TrainAcc = 0
    ValAcc = 0

    for (x, y) in _trainDataLoader:
        (x, y) = (x.to(device), y.to(device))

        pred = model(x)
        loss = lossFunction(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        TrainLoss += loss
        TrainAcc += (pred.argmax(1) == y).type(torch.float).sum().item()

    with torch.no_grad():
        model.eval()

        for (x, y) in _valDataLoader:
            (x, y) = (x.to(device), y.to(device))

            pred = model(x)
            ValLoss += lossFunction(pred, y)

            ValAcc += (pred.argmax(1) == y).type(torch.float).sum().item()


    avgTrainLoss = TrainLoss / _trainStep
    avgValLoss = ValLoss / _valStep

    TrainAcc /= len(_trainDataLoader.dataset)
    ValAcc /= len(_valDataLoader.dataset)

    History["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    History["train_acc"].append(TrainAcc)
    History["val_loss"].append(avgValLoss.cpu().detach().numpy())
    History["val_acc"].append(ValAcc)

end_time = time.time()

print(f'Time taken to train model {(end_time - start_time)}')

with torch.no_grad():
    model.eval()

    pred_res = []

    for (x, y) in _testDataLoader:
        x = torchvision.transforms.functional.resize(x, [224, 224, 3])
        x = x.to(device)

        pred = model(x)
        pred_res.extend(pred.argmax(axis=1).cpu().numpy())

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