from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import ReLU
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import LogSoftmax
from torch import flatten

class cnn(Module):
    def __init__(self, numOfChannels, classes):
        super(cnn).__init__()

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