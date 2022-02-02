import cv2
import pandas as pd
from scipy.ndimage import convolve
import numpy as np
import matplotlib.pyplot as plt
from util import *

WINDOW_PARAM = 10
SIGMA = 1


def main():
    img = cv2.imread("picassoBlurImage.png", cv2.IMREAD_GRAYSCALE) / 255.
    latent = cv2.imread("picassoOut.png", cv2.IMREAD_GRAYSCALE) / 255.
    # padding(img, 100)
    # computOmega(img, 10)
    omega = computOmega(img, WINDOW_PARAM)
    print(computeLocalPrior(latent, img, omega, SIGMA))


if __name__ == '__main__':
    main()
