import cv2
import pandas as pd
from scipy.ndimage import convolve
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from update import *
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

    omega = computOmega(img, WINDOW_PARAM)

    cur = time.time()
    temp = np.random.uniform(low=0, high=0.5, size=(11, 11))
    temp[5][5] = 1

    deblurImage(img, latent, omega)

    lapsed = (time.time() - cur)
    print(lapsed)

    np.savetxt('sampleLA.csv', latent, delimiter=",")


if __name__ == '__main__':
    main()
