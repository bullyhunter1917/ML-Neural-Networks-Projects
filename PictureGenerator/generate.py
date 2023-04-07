import torch
import cv2
import torchvision
from PIL import Image

from u_net import U_net
from DDPM import Diffusion

def save_images(images, path):
    grid = torchvision.utils.make_grid(images)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

SIZE = 32
DEVICE = 'cuda'

model = U_net(SIZE).to(DEVICE)
model.load_state_dict(torch.load('output/models/model.pt'))

sampler = Diffusion(imsize=SIZE)

picture = sampler.sample(model, 1)
picture = torchvision.transforms.Resize(256)(picture)

save_images(picture, f'output/GeneratedPics/{SIZE}.jpg')
