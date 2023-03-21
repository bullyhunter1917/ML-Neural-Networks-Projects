from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten

class LeNet(Module):
    #Defining how neural network will look and work
    def __init__(self, numOfChannels, classes):
        super(LeNet, self).__init__()

        self.conv1 = Conv2d(in_channels=numOfChannels, out_channels=20, kernel_size=(5,5))
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv2 = Conv2d(in_channels= 20,out_channels= 50,kernel_size=(5,5))
        self.relu2 = ReLU()
        self.pool2 = MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.linear = Linear(in_features=800, out_features=500)
        self.relu3 = ReLU()

        self.linear2 = Linear(in_features=500, out_features=classes)
        self.softmax = LogSoftmax(dim=1)

    #Creating forward pass
    def forward(self, x):
        #First filter
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        #Second filter
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        #Fully connected network
        x = flatten(x, 1)
        x = self.linear(x)
        x = self.relu3(x)

        x = self.linear2(x)
        output = self.softmax(x)

        return output