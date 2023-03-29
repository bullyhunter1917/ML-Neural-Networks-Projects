import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

BETA_start = 0.0001
BETA_end = 0.02
T = 1000

beta_t = np.linspace(BETA_start, BETA_end, T)
a_t = np.linspace((1 - BETA_start), (1 - BETA_end), 1000)


# Whole forward in one step
def forward(x):
    return np.sqrt(np.prod(a_t))*x + np.sqrt(1 - np.prod(a_t))*np.random.normal(0, 1, x.shape)


# One forward step
def forward_t(x, t):
    return np.sqrt(np.prod(a_t[:t]))*x + np.sqrt(1 - np.prod(a_t[:t]))*np.random.normal(0, 1, x.shape)


def sampling():
    xt = np.random.normal(0, 1)
    for i in range(T, 1, -1):
        if i > 1:
            z = np.random.normal(0, 1)
        else:
            z = 0

        xt = 1/np.sqrt(a_t[i])*(xt - (1 - a_t[i])/np.sqrt(1 - np.prod(a_t[:i]))* epsilon_theta ) + sigma_t*z

    return xt


def train():
    x0 = sampling()
    t = np.random.uniform(0, T)
    epsilon = np.random.normal(0, 1)

    # Policzyć gradient????

    return


originalPicture = plt.imread('Mallard.jpg')

# Normalizing picture
originalPicture = originalPicture.astype(float)
originalPicture /= 255

plt.imshow(originalPicture)
plt.show()

plt.imshow(forward_t(originalPicture, 300))
plt.show()