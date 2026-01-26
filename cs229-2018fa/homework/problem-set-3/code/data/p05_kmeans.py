from re import X
import matplotlib.pyplot as plt
from matplotlib.image import imread 
import numpy as np
import os
import sys
import random

PLOT_COLORS = ['red', 'green', 'blue', 'orange']
K = 16
NUM_TRIALS = 2

def find_nearest(target, mus):
    dists = np.linalg.norm(mus - target, axis=1)
    return np.argmin(dists)

def main(image_path,max_iter=30):
    A = imread(image_path)
    H, W = A.shape[:2]

    pixels = A.reshape(-1, 3).astype(float)

    mu = np.zeros((K, 3))
    for i in range(K):
        x = np.random.randint(0, H)
        y = np.random.randint(0, W)
        mu[i] = pixels[x*W + y]

    eps = 1e10
    it = 0

    while it < max_iter and eps > 1e-1:
        c = np.array([find_nearest(p, mu) for p in pixels])

        new_mu = np.zeros_like(mu)
        for i in range(K):
            mask = (c == i)
            if np.sum(mask) > 0:
                new_mu[i] = pixels[mask].mean(axis=0)
            else:
                new_mu[i] = mu[i]

        eps = np.linalg.norm(new_mu - mu)
        mu = new_mu
        it += 1

    clustered_pixels = mu[c].reshape(H, W, 3).astype(np.uint8)
    plt.imshow(clustered_pixels)
    plt.axis('off')
    plt.show()

    plt.imsave('pepper-condensed.tiff', clustered_pixels)
    return clustered_pixels
if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    image_path = sys.argv[1]
    for t in range(NUM_TRIALS):
        main(image_path)

