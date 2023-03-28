import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def forward(x, mean, sigma):
    return x + (255*np.random.normal(mean, sigma, x.shape)).astype(int)

originalPicture = plt.imread('Mallard.jpg')




plt.imshow(forward(originalPicture, 0, 0.5))
plt.show()