from cv2 import BORDER_REFLECT, IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_UNCHANGED
import pandas as pd
from scipy.ndimage import convolve
import scipy.stats
import cv2
import numpy as np
import matplotlib.pyplot as plt


def padding(img, window_param):
    temp = cv2.copyMakeBorder(img, window_param, window_param,
                              window_param, window_param, cv2.BORDER_REFLECT)

    return temp


def computOmega(img, window_param):
    padded = padding(img, window_param)
    res = np.zeros(shape=img.shape)
    for i in np.arange(window_param, img.shape[0]+window_param):
        for j in np.arange(window_param, img.shape[1]+window_param):
            temp_img = padded[i:i+window_param*2+1, j:j+window_param*2+1]
            st = np.std(temp_img)
            if st < 5/255.:
                res[i-window_param][j-window_param] = 1
    return res
    # res = np.zeros(
    #     shape=(img.shape[0]-window_param * 2, img.shape[1]-window_param*2))
    # for i in np.arange(window_param, img.shape[0]-window_param):
    #     for j in np.arange(window_param, img.shape[1]-window_param):
    #         temp_img = img[i:i+window_param*2+1, j:j+window_param*2+1]
    #         st = np.std(temp_img)
    #         if st < 5:
    #             res[i-window_param][j-window_param] = 1

    # cv2.imshow("original", img)
    # cv2.imshow("omega", res)
    # cv2.imshow("pad", padded)
    # cv2.imshow("plus", img+res)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # print(temp_img.shape)
    # plt.figure(figsize=(10, 5))
    # plt.subplot(121), plt.imshow(img, cmap='gray')
    # plt.title("Input Image"), plt.axis('off')
    # plt.subplot(122), plt.imshow(res, cmap='gray')
    # plt.title("Omega"), plt.axis('off')
    # plt.show()


def computeLocalPrior(latent, img, omega, sigma=1):
    ret = 1

    print("Original Image: ", img.shape)
    print("Latent Image", img.shape)
    print("Omega Size: ", omega.shape)
    gaus = scipy.stats.norm(0, sigma)

    for i in range(len(latent)-1):
        for j in range(len(latent[i])-1):
            if omega[i][j] == 1:
                grad_x = (latent[i+1][j] - latent[i][j]) - \
                    (img[i+1][j] - img[i][j])
                grad_y = (latent[i][j+1] - latent[i][j]) - \
                    (img[i][j+1] - img[i][j])
                gaus_x = gaus.cdf(grad_x)
                gaus_y = gaus.cdf(grad_y)
                ret *= gaus_x * gaus_y
                if gaus_x == 0 or gaus_y == 0:
                    print((i, j))

    return ret
