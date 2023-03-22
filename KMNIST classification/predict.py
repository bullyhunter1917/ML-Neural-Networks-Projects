import argparse
import imutils as imutils
import numpy as np
import torch
import torchvision.datasets
import cv2

from torch.utils.data import Subset, DataLoader
from torchvision.transforms import ToTensor

argparser = argparse.ArgumentParser()
argparser.add_argument("-m", "--model", type=str, required=True, help="path to the trained PyTorch model")
args = vars(argparser.parse_args())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading KMNIST and choosing 10 random pictures from test
_test = torchvision.datasets.KMNIST(root='data', train=False, download=True, transform=ToTensor())
indexes = np.random.choice(range(0, len(_test)), size=(9,))
_test = Subset(_test, indexes)

# Creating data loader
_testDataLoader = DataLoader(_test, batch_size=1)

# Loading model from file
model = torch.load(args["model"]).to(device)

result_picture = np.zeros((384, 384, 3))

# Classifying pictures and displaying them
with torch.no_grad():
    model.eval()

    pictureIndex = 0

    for (image, label) in _testDataLoader:
        original_img = image.numpy().squeeze(axis=(0, 1))
        properLabel = _test.dataset.classes[label.numpy()[0]]

        image = image.to(device)
        pred = model(image)

        index = pred.argmax(axis=1).cpu().numpy()[0]
        predLabel = _test.dataset.classes[index]

        # From gray scale to RGB
        original_img = np.dstack([original_img] * 3)
        original_img = imutils.resize(original_img, width=128)

        # Adding proper label with prediction result
        color = (0, 255, 0) if properLabel == predLabel else (0, 0, 255)
        cv2.putText(original_img, properLabel, (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)

        # Putting picture into photo collage
        result_picture[128*(pictureIndex//3):128*(pictureIndex//3 + 1), 128*(pictureIndex%3):128*(pictureIndex%3 + 1)] = original_img

        pictureIndex += 1

# Showing result
cv2.imshow('image', result_picture)
cv2.waitKey(0)

exit()
