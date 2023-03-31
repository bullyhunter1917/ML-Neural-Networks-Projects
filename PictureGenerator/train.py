import os.path

import torch
import torchvision.utils
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import torchvision.transforms as trans
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import DDPM
import u_net
from PIL import Image

def save_images(images, path):
    grid = torchvision.utils.make_grid(images)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SIZE = 256
EPOCH = 360
BATCH_SIZE = 12
NOISE_SIZE = 1000
LR = 3e-4


_train = CIFAR10(root='data', train=True, download=True, transform=trans.Compose(
    [trans.Resize((SIZE, SIZE)),
    trans.ToTensor()]
))

_test = CIFAR10(root='data', train=False, download=True, transform=trans.Compose(
    [trans.Resize((SIZE, SIZE)),
    trans.ToTensor()]
))

_trainDataLoader = DataLoader(_train, BATCH_SIZE, shuffle=True)
_testDataLoader = DataLoader(_test, BATCH_SIZE)

sampling = DDPM.Diffusion()
model = u_net.U_net().to(DEVICE)
optimaizer = AdamW(params=model.parameters(), lr=LR)
mse = nn.MSELoss()

for i in tqdm(range(EPOCH)):
    model.train()

    for (x, y) in _trainDataLoader:
        (x, y) = (x.to(DEVICE), y.to(DEVICE))
        t = sampling.sample_timestamp(BATCH_SIZE)
        xt, noise = sampling.noise_image(x, t)
        predicted_noise = model(xt, t)
        loss = mse(predicted_noise, noise)

        optimaizer.zero_grad()
        loss.back()
        optimaizer.step()

    sampled_images = sampling.sample(model, n=BATCH_SIZE)
    save_images(sampled_images, f'output/pictures/{i}.jpg')
    torch.save(model.state_dict(), 'output/models/model.pt')
