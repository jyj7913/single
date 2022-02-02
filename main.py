import cv2
import pandas as pd
from scipy.ndimage import convolve
import numpy as np
import matplotlib.pyplot as plt
from util import *


def main():
    img = cv2.imread("picassoBlurImage.png", cv2.IMREAD_GRAYSCALE)
    # padding(img, 100)
    computOmega(img, 10)


if __name__ == '__main__':
    main()
