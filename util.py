from cv2 import BORDER_REFLECT, IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_UNCHANGED
import pandas as pd
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
from update import *

LAMBDA1 = 0.1
LAMBDA2 = 15
GAMMA = 8


def psf2otf(flt, img_shape):
    flt_top_half = flt.shape[0]//2
    flt_bottom_half = flt.shape[0] - flt_top_half
    flt_left_half = flt.shape[1]//2
    flt_right_half = flt.shape[1] - flt_left_half
    flt_padded = np.zeros(img_shape, dtype=flt.dtype)
    flt_padded[:flt_bottom_half,
               :flt_right_half] = flt[flt_top_half:, flt_left_half:]
    flt_padded[:flt_bottom_half, img_shape[1] -
               flt_left_half:] = flt[flt_top_half:, :flt_left_half]
    flt_padded[img_shape[0]-flt_top_half:,
               :flt_right_half] = flt[:flt_top_half, flt_left_half:]
    flt_padded[img_shape[0]-flt_top_half:, img_shape[1] -
               flt_left_half:] = flt[:flt_top_half, :flt_left_half]
    return np.fft.fft2(flt_padded)


def conv2dn(a, f):
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape=s, strides=a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)


def conv2dt(a, f):
    ret = F.conv2d(Variable(a.view(1, 1, a.shape[0], a.shape[1])),
                   Variable(f.view(1, 1, f.shape[0], f.shape[1])))
    ret.squeeze_()
    return ret


def padding(img, window_param):
    ret = cv2.copyMakeBorder(
        img, window_param, window_param, window_param, window_param, cv2.BORDER_REFLECT)
    return ret


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


def psi_x(psi, a):
    energy = LAMBDA1 * abs(eq(psi))
    energy += LAMBDA2 * a[2] * ((psi - a[0])**2)
    energy += GAMMA * ((psi - a[1])**2)
    return energy


def psi_y(psi, a):
    energy = LAMBDA1 * abs(eq(psi))
    energy += LAMBDA2 * a[2] * ((psi - a[0])**2)
    energy += GAMMA * ((psi - a[1])**2)
    return energy


def trii(img_shape):
    x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    fx = psf2otf(x, img_shape)
    fy = psf2otf(y, img_shape)

    fxx = fx * fx
    fyy = fy * fy
    fxy = fx * fy

    xout = (50/(2**1)) * (fx * np.conjugate(fx))
    yout = (50/(2**1)) * (fy * np.conjugate(fy))
    xxout = (50/(2**2)) * (fxx * np.conjugate(fxx))
    yyout = (50/(2**2)) * (fyy * np.conjugate(fyy))
    xyout = (50/(2**2)) * (fxy * np.conjugate(fxy))

    return xout + yout + xxout + yyout + xyout


def eq(x):
    if abs(x) > 10:
        return -(6.1e-4 * (x**2) + 5.0)
    else:
        return -2.7 * abs(x)


def computeGlobalPrior(latent):
    for i in range(len(latent)-1):
        for j in range(len(latent[i])-1):
            print(1)
