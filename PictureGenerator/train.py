import torch
import torchvision.utils
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import torchvision.transforms as trans
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
import DDPM
from u_net import U_net
from PIL import Image
import cv2

def save_images(images, path):
    grid = torchvision.utils.make_grid(images)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


DEVICE = 'cuda'
SIZE = 32
EPOCH = 500
BATCH_SIZE = 12
NOISE_SIZE = 1000
LR = 3e-4


_train = CIFAR10(root='data', train=True, download=True, transform=trans.Compose([
    trans.Resize((SIZE, SIZE)),
    trans.ToTensor()
]))

_train = Subset(_train, range(120))

# _test = CIFAR10(root='data', train=False, download=True, transform=trans.Compose(
#     [trans.Resize((SIZE, SIZE)),
#     trans.ToTensor()]
# ))


_trainDataLoader = DataLoader(_train, BATCH_SIZE, shuffle=True)
# _testDataLoader = DataLoader(_test, BATCH_SIZE)

model = U_net(SIZE).to(DEVICE)
optimaizer = AdamW(params=model.parameters(), lr=LR)
mse = nn.MSELoss()
sampling = DDPM.Diffusion(imsize=SIZE, device=DEVICE)

for epoch in range(EPOCH):
    pbar = tqdm(_trainDataLoader)

    for j, (x, _) in enumerate(pbar):
        print(x.shape)
        pic = x.permute(0, 2, 3, 1).numpy().squeeze()
        pic = cv2.cvtColor(pic[0], cv2.COLOR_BGR2RGB)
        cv2.imshow('cos', pic)
        cv2.waitKey(0)
        print(f'SIZE: {len(x)}')
        x = x.to(DEVICE)
        t = sampling.sample_timestamp(BATCH_SIZE).to(DEVICE)
        xt, noise = sampling.noise_image(x, t)
        predicted_noise = model(xt, t)
        loss = mse(predicted_noise, noise)

        optimaizer.zero_grad()
        loss.backward()
        optimaizer.step()

    sampled_images = sampling.sample(model, n=BATCH_SIZE)
    save_images(sampled_images, f'output/pictures/{epoch}.jpg')
    torch.save(model.state_dict(), 'output/models/model.pt')
