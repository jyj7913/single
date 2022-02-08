import cv2
import pandas as pd
from scipy.ndimage import convolve
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from util import *
import csv
import time

from scipy.optimize import minimize_scalar as mins

WINDOW_PARAM = 10
SIGMA = 0.5


def main():

    img = cv2.imread("dataset/picassoBlurImage.png",
                     cv2.IMREAD_GRAYSCALE) / 255.
    latent = cv2.imread("dataset/picassoBlurImage.png",
                        cv2.IMREAD_GRAYSCALE) / 255.

    # omega = computOmega(img, WINDOW_PARAM)

    # psi_xx = np.genfromtxt('sample00.csv', delimiter=',')
    # psi_yy = np.genfromtxt('sample01.csv', delimiter=',')

    cur = time.time()
    temp = np.random.uniform(low=0, high=0.5, size=(49, 49))
    temp[24][24] = 1
    optimizeF(img, latent, psf=temp)

    lapsed = (time.time() - cur)
    print(lapsed)

    np.savetxt('sampleLA.csv', latent, delimiter=",")


if __name__ == '__main__':
    main()
