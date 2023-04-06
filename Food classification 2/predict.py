import torchvision
from torchvision.datasets import Food101
import numpy as np
import cv2
import torch.utils.data
import imutils as imutils

from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import ReLU
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import LogSoftmax
from torch import flatten

class Cnn(Module):
    def __init__(self, numOfChannels, classes):
        super(Cnn, self).__init__()

        self.conv1_1 = Conv2d(in_channels=numOfChannels, out_channels=64, kernel_size=(3, 3), padding=1)
        self.relu = ReLU()
        self.conv1_2 = Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1)
        self.maxpool = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2_1 = Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
        self.conv2_2 = Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1)

        self.conv3_1 = Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1)
        self.conv3_2 = Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1)
        self.conv3_3 = Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1)

        self.conv4_1 = Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1)
        self.conv4_2 = Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)
        self.conv4_3 = Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)

        self.conv5_1 = Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)
        self.conv5_2 = Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)
        self.conv5_3 = Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)

        self.l1 = Linear(in_features=25088, out_features=4096)
        self.l2 = Linear(in_features=4096, out_features=4096)
        self.l3 = Linear(in_features=4096, out_features=classes)
        self.softmax = LogSoftmax(dim=1)

    def forward(self, x):
        # First layer
        x = self.conv1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Second layer
        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Third layer
        x = self.conv3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.relu(x)
        x = self.conv3_3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Fourth layer
        x = self.conv4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.relu(x)
        x = self.conv4_3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Fifth layer
        x = self.conv5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.relu(x)
        x = self.conv5_3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Fully connected network
        x = flatten(x, 1)
        x = self.l1(x)
        x = self.relu(x)

        x = self.l2(x)
        x = self.relu(x)

        x = self.l3(x)
        output = self.softmax(x)

        return output

output_img_size = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


_test = Food101(root='data', split='test', download=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor()
]))

indexes = np.random.choice(range(0, len(_test)), size=(9,))
_test = torch.utils.data.Subset(_test, indexes)

_testDataLoader = torch.utils.data.DataLoader(dataset=_test, batch_size=1)

model = torch.load('output/model.pth').to(device)

result_picture = np.zeros((672, 672, 3))

with torch.no_grad():
    model.eval()

    pictureIndex = 0

    for (image, label) in _testDataLoader:
        original_img = image.permute(0, 2, 3, 1).numpy().squeeze()
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        properLabel = _test.dataset.classes[label.numpy()[0]]

        image = image.to(device)
        pred = model(image)

        index = pred.argmax(axis=1).cpu().numpy()[0]
        predLabel = _test.dataset.classes[index]

        print(predLabel)

        # Adding proper label with prediction result
        color = (0, 255, 0) if properLabel == predLabel else (0, 0, 255)
        original_img = cv2.putText(original_img, properLabel, (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2, cv2.LINE_AA)

        # Putting picture into photo collage
        result_picture[output_img_size * (pictureIndex // 3):output_img_size * (pictureIndex // 3 + 1),
        output_img_size * (pictureIndex % 3):output_img_size * (pictureIndex % 3 + 1)] = original_img

        pictureIndex += 1

    # Showing result
    cv2.imshow('image', result_picture)
    cv2.waitKey(0)

    exit()