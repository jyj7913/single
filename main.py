import cv2
import pandas as pd
from scipy.ndimage import convolve
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from util import *
import csv

WINDOW_PARAM = 10
SIGMA = 0.5


def main():
    img = cv2.imread("dataset/picassoBlurImage.png",
                     cv2.IMREAD_GRAYSCALE) / 255.
    latent = cv2.imread("dataset/picassoOut.png", cv2.IMREAD_GRAYSCALE) / 255.

    omega = computOmega(img, WINDOW_PARAM)
    a = computeLocalPrior(latent, img, omega, SIGMA)
    print(a)


if __name__ == '__main__':
    main()
