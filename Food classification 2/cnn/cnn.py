from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import ReLU
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import LogSoftmax


class cnn(Module):
    def __init__(self, numOfChannels, classes):
        super(cnn).__init__()

        self.conv1_1 = Conv2d(in_channels=numOfChannels, out_channels=64, kernel_size=(3, 3), padding=1)
        self.relu = ReLU()
        self.conv1_2 = Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1)
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2_1 = Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
        self.conv2_2 = Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1)
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3_1 = Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1)
        self.conv3_2 = Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1)
        self.conv3_3 = Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1)
        self.maxpool3 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv4_1 = Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1)
        self.conv4_2 = Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)
        self.conv4_3 = Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)
        self.maxpool4 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv5_1 = Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)
        self.conv5_2 = Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)
        self.conv5_3 = Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)
        self.maxpool5 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.l1 = Linear(in_features=4096, out_features=4096)
        self.l2 = Linear(in_features=4096, out_features=4096)
        self.l3 = Linear(in_features=4096, out_features=classes)
        self.softmax = LogSoftmax(dim=1)
