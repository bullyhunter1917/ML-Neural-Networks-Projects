import torch
import torch.nn as nn
import torch.nn.functional as F

class U_net(nn.Module):
    def __int__(self, inChannels=3, outChannels=3, time_dim=256, device='cuda'):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        # 256x256x3

        self.input = DoubleConv(inChannels, 32, )

        # Encoder layer
        self.down1 = Down(32, 64)
        self.sa1 = SelfAttention(64, 128) # 128x128x64
        self.down2 = Down(64, 128)
        self.sa2 = SelfAttention(128, 64) # 64x64x128
        self.down3 = Down(128, 256)
        self.sa3 = SelfAttention(256, 32) # 32x32x256
        self.down4 = Down(256, 512)
        self.sa4 = SelfAttention(512, 16) # 16x16x512

        # Bottom layer
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2_1 = nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=1024, padding=1)
        self.conv2_2 = nn.Conv2d(kernel_size=(3, 3), in_channels=1024, out_channels=1024, padding=1)

        # Decoder layer
        self.up1 = Up(1024, 512)
        self.sa5 = SelfAttention(512, 16)
        self.up2 = Up(512, 256)
        self.sa6 = SelfAttention(256, 32)
        self.up3 = Up(256, 128)
        self.sa7 = SelfAttention(128, 64)
        self.up4 = Up(128, 64)
        self.sa8 = SelfAttention(64, 128)
        self.up5 = Up(64, 32)
        self.sa9 = SelfAttention(32, 256)

        self.output = Conv2d(kernel_size=(1, 1), in_channels=32, out_channels=outChannels)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device))
        )
        pos_enc_a = torch.sin(t.repeat(1, channels//2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels//2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=1)

        return pos_enc


    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.conv1_1(x)
        x1 = self.relu(x1)
        x1 = self.conv1_2(x1)
        x1 = self.relu(x1)

        # Encoder layer
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)

        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)

        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x5 = self.down4(x4, t)
        x5 = self.sa4(x5)

        # Bottom layer
        x6 = self.maxpool(x5)
        x6 = self.conv2_1(x6)
        x6 = self.conv2_2(x6)

        # Decoder layer
        x = self.up1(x6, x5, t)
        x = self.sa5(x)

        x = self.up2(x, x4, t)
        x = self.sa6(x)

        x = self.up3(x, x3, t)
        x = self.sa7(x)

        x = self.up4(x, x2, t)
        x = self.sa8(x)

        x = self.up5(x, x1, t)
        x = self.sa9(x)

        out = self.output(x)

        return out


class DoubleConv(nn.Module):
    def __init__(self, inChannels, outChannels, midChannels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not midChannels:
            midChannels = outChannels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=inChannels, out_channels=midChannels, kernel_size=(3, 3), padding=1, bias=False),
            nn.GroupNorm(1, midChannels),
            nn.GELU(),
            nn.Conv2d(in_channels=midChannels, out_channels=outChannels, kernel_size=(3, 3), padding=1, bias=False),
            nn.GroupNorm(1, outChannels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __int__(self, inChannels, outChannels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)),
            DoubleConv(inChannels, inChannels, residual=True),
            DoubleConv(inChannels, outChannels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                outChannels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

        return x + emb

class Up(nn.Module):
    def __int__(self, inChannels, outChannels, emb_dim=256):
        self.upPool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        self.convUp = nn.Sequential(
            DoubleConv(inChannels, inChannels, residual=True),
            DoubleConv(inChannels, outChannels, inChannels//2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                outChannels
            ),
        )

    def forward(self, x, x_skip, t):
        x = self.upPool(x)
        x = torch.cat([x_skip, x], dim=1)
        x = self.convUp(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class SelfAttention(nn.Module):
    def __int__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size

        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)