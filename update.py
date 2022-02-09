from scipy.ndimage import convolve
import scipy.stats
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar as mins
from scipy.optimize import minimize as minn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from util import *

LAMBDA1 = 0.1
LAMBDA2 = 15
GAMMA = 8


def computePsi(img, latent, omega):
    pad_img = padding(img, 1)
    pad_lat = padding(latent, 1)
    temp = np.zeros(shape=(2, img.shape[0], img.shape[1]))

    img_x = conv2dn(pad_img, np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))
    img_y = conv2dn(pad_img, np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))
    lat_x = conv2dn(pad_lat, np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))
    lat_y = conv2dn(pad_lat, np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))

    for i in range(len(img)):
        for j in range(len(img[i])):
            res_x = mins(psi_x, None, None, [
                         img_x[i][j], lat_x[i][j], omega[i][j]])
            temp[0][i][j] = res_x.x

            res_y = mins(psi_y, None, None, [
                         img_y[i][j], lat_y[i][j], omega[i][j]])
            temp[1][i][j] = res_y.x
    return temp


def computePsi2(img, latent, omega, psi):
    m = psiModel2(psi, img, latent, omega)
    optim = torch.optim.Adam(m.parameters(), lr=0.001)
    ret = training_loop_psi(m, optim)
    return ret


def computeLatent(img, psi, psf):
    tri = trii(img.shape)
    sx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    fsx = psf2otf(sx, img.shape)
    fsy = psf2otf(sy, img.shape)
    cfsx = np.conjugate(fsx)
    cfsy = np.conjugate(fsy)

    fpsf = psf2otf(psf, img.shape)
    cfpsf = np.conjugate(fpsf)

    fimg = np.fft.fft2(img)

    son = np.multiply(cfpsf, fimg)
    son = np.multiply(son, tri)

    fpsix = np.fft.fft2(psi[0])
    fpsiy = np.fft.fft2(psi[1])
    son += GAMMA * cfsx * fpsix
    son += GAMMA * cfsy * fpsiy

    down = cfpsf * fpsf
    down = down * tri

    down += GAMMA * np.conjugate(fsx) * fsx
    down += GAMMA * np.conjugate(fsy) * fsy
    down += 0.0001

    ret = np.divide(son, down)
    ret = np.fft.ifft2(ret)
    ret = np.abs(ret)

    return ret


def optimizeL(img, latent, omega, psi, psf):
    for i in range(15):
        # psi = computePsi(img, latent, omega)
        latent = computeLatent(img, psi, psf)
        psi = computePsi2(img, latent, omega, psi)
    return latent


def optimizeF(img, latent, omega=None, psi=None, psf=None):
    m = psfModel(psf, img, latent)
    optim = torch.optim.Adam(m.parameters(), lr=0.001)
    ret = training_loop(m, optim)
    return ret


def solveF(psf, arg):
    psf = psf.reshape(41, 41)
    sum = 0
    for i in range(5):
        temp = conv2dn(arg[1][i], psf) - arg[0][i]
        temp = np.linalg.norm(temp).item() ** 2
        temp *= 50 / (2 ** arg[2][i])
        sum += temp
    sum += np.sum(np.abs(psf))
    return sum


def deblurImage(img, latent, omega):
    psi = (latent, latent)
    psf = np.random.uniform(low=0, high=0.5, size=(11, 11))
    psf[5][5] = 1

    for i in range(15):
        optimizeL(img, latent, omega, psi, psf)
        optimizeF(img, latent, pfs=psf)
