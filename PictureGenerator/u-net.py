from torch.nn import Module
from torch.nn import Conv2d, ReLU, MaxPool2d, Upsample
from torch.nn import Sequential
class U_net(Module):
    def __int__(self, inChannels=3, outChannels=3, device='cuda'):
        super.__init__()
        self.device = device

        # 256x256x3

        self.conv1_1 = Conv2d(kernel_size=(3, 3), in_channels=inChannels, out_channels=32, padding=1)
        self.conv1_2 = Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=32, padding=1)
        self.relu = ReLU()

        self.down1 = Down(32, 64)
        self.sa1 = SelfAttention(64, 128)
        self.down2 = Down(64, 128)
        self.sa2 = SelfAttention(128, 64)
        self.down3 = Down(128, 256)
        self.sa3 = SelfAttention(256, 32)
        self.down4 = Down(256, 512)
        self.sa4 = SelfAttention(512, 16)

        



class Down(Module):
    def __int__(self, inChannels, outChannels):
        self.maxPool = MaxPool2d(kernel_size=(2, 2))
        self.convDown = Sequential(
            Conv2d(kernel_size=(3, 3), in_channels=inChannels, out_channels=outChannels, padding=1),
            ReLU(),
            Conv2d(kernel_size=(3, 3), in_channels=outChannels, out_channels=outChannels, padding=1),
            ReLU()
        )

class Up(Module):
    def __int__(self, inChannels, outChannels):
        self.upPool = Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        self.convUp = Sequential(


        )


class SelfAttention(Module):
    def __int__(self):
        self.cos = 1