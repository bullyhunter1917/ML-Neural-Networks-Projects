import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

BETA_start = 0.0001
BETA_end = 0.02
T = 1000

beta_t = np.linspace(BETA_start, BETA_end, T)
one_minus_beta_t = np.linspace((1 - BETA_start), (1 - BETA_end), 1000)


# Whole forward in one step
def forward(x):
    return np.sqrt(np.prod(one_minus_beta_t))*x + np.sqrt(1 - np.prod(one_minus_beta_t))*np.random.normal(0, 1, x.shape)


# One forward step
def forward_one_step(x, t):
    return np.sqrt(one_minus_beta_t[t])*x + np.sqrt(beta_t[t])*np.random.normal(0, 1, x.shape)


originalPicture = plt.imread('Mallard.jpg')

# Normalizing picture
originalPicture = originalPicture.astype(float)
originalPicture /= 255
