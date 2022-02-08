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

LAMBDA1 = 0.1
LAMBDA2 = 15
GAMMA = 8


class psfModel(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """

    def __init__(self, psf, img, latent):
        super().__init__()
        # initialize weights with random numbers
        # weights = torch.distributions.Uniform(0, 0.1).sample((3,))
        weights = psf

        pad_img = padding(img, 1)
        pad_lat = padding(latent, 1)
        sx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        img_x = conv2dn(pad_img, sx)
        img_y = conv2dn(pad_img, sy)
        pad_img_x = padding(img_x, 1)
        pad_img_y = padding(img_y, 1)
        img_xx = conv2dn(pad_img_x, sx)
        img_xy = conv2dn(pad_img_x, sy)
        img_yy = conv2dn(pad_img_y, sy)
        self.img_grad = [torch.Tensor(img), torch.Tensor(img_x), torch.Tensor(
            img_y), torch.Tensor(img_xx), torch.Tensor(img_xy), torch.Tensor(img_yy)]

        lat_x = conv2dn(pad_lat, sx)
        lat_y = conv2dn(pad_lat, sy)
        pad_lat_x = padding(lat_x, 1)
        pad_lat_y = padding(lat_y, 1)
        lat_xx = conv2dn(pad_lat_x, sx)
        lat_xy = conv2dn(pad_lat_x, sy)
        lat_yy = conv2dn(pad_lat_y, sy)
        self.lat_grad = [torch.Tensor(padding(latent, psf.shape[0]//2)), torch.Tensor(padding(lat_x, psf.shape[0]//2)), torch.Tensor(padding(lat_y, psf.shape[0]//2)),
                         torch.Tensor(padding(lat_xx, psf.shape[0]//2)), torch.Tensor(padding(lat_xy, psf.shape[0]//2)), torch.Tensor(padding(lat_yy, psf.shape[0]//2))]

        self.w = [0, 1, 1, 2, 2, 2]

        # make weights torch parameters
        self.weights = nn.Parameter(torch.Tensor(weights), requires_grad=True)

    def forward(self):
        """Implement function to be optimised. In this case, an exponential decay
        function (a + exp(-k * X) + b),
        """
        psf = self.weights
        sum = 0
        for i in range(5):
            temp = conv2dt(self.lat_grad[i], psf) - self.img_grad[i]
            temp = torch.norm(temp).item() ** 2
            temp *= 50 / (2 ** self.w[i])
            sum += temp
        sum += torch.norm(psf, p=1).item()

        return sum


def training_loop(model, optimizer, n=15):
    "Training loop for torch model."
    losses = []
    loss_fn = nn.MSELoss()
    for i in range(n):
        preds = model()
        loss = loss_fn(torch.Tensor([preds]), torch.Tensor([0]))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return model.parameters()


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
    arr = []

    for i in range(len(latent)-1):
        for j in range(len(latent[i])-1):
            if omega[i][j] == 1:
                grad_x = (latent[i+1][j] - latent[i][j]) - \
                    (img[i+1][j] - img[i][j])
                grad_y = (latent[i][j+1] - latent[i][j]) - \
                    (img[i][j+1] - img[i][j])
                gaus_x = gaus.pdf(grad_x)
                gaus_y = gaus.pdf(grad_y)
                ret *= gaus_x * gaus_y
                arr.append(ret)

    return arr


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


def computePsi(img, latent, omega):
    energy = 0

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


def trii():
    x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    la = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    fx = np.fft.fft2(x)
    fy = np.fft.fft2(y)
    fla = np.fft.fft2(la)
    xx = (50/(2**1)) * (np.multiply(np.conjugate(fx), fx))
    yy = (50/(2**1)) * (np.multiply(np.conjugate(fy), fy))
    ll = (50/(2**2)) * (np.multiply(np.conjugate(fla), fla))
    return xx + yy + ll + ll + ll + ll


def computeLatent(img, psi, psf):
    tri = trii()
    fsx = np.fft.fft2(np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))
    fsy = np.fft.fft2(np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))

    fpsf = np.fft.fft2(psf)
    cfpsf = np.conjugate(fpsf)
    fimg = np.fft.fft2(img)

    son = np.multiply(cfpsf, fimg)
    son = np.multiply(son, tri)

    fpsix = np.fft.fft2(psi[0])
    fpsiy = np.fft.fft2(psi[1])
    son += GAMMA * np.multiply(np.conjugate(fsx), fpsix)
    son += GAMMA * np.multiply(np.conjugate(fsy), fpsiy)

    down = np.multiply(cfpsf, fpsf)
    down = np.multiply(down, tri)

    down += GAMMA * np.multiply(np.conjugate(fsx), fsx)
    down += GAMMA * np.multiply(np.conjugate(fsy), fsy)

    ret = np.divide(son, down)
    ret = np.fft.ifft2(ret)

    return ret


def optimizeL(img, latent, omega, psi, psf):
    for i in range(15):
        psi = computePsi(img, latent, omega)
        latent = computeLatent(img, psi, psf)
    return latent


def optimizeF(img, latent, omega=None, psi=None, psf=None):
    # pad_img = padding(img, 1)
    # pad_lat = padding(latent, 1)
    # temp = np.zeros(shape=(2, img.shape[0], img.shape[1]))

    # sx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    # sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # img_x = conv2dn(pad_img, sx)
    # img_y = conv2dn(pad_img, sy)

    # pad_img_x = padding(img_x, 1)
    # pad_img_y = padding(img_y, 1)
    # img_xx = conv2dn(pad_img_x, sx)
    # img_xy = conv2dn(pad_img_x, sy)
    # img_yy = conv2dn(pad_img_y, sy)
    # img_grad = [img, img_x, img_y, img_xx, img_xy, img_yy]

    # lat_x = conv2dn(pad_lat, sx)
    # lat_y = conv2dn(pad_lat, sy)
    # pad_lat_x = padding(lat_x, 1)
    # pad_lat_y = padding(lat_y, 1)
    # lat_xx = conv2dn(pad_lat_x, sx)
    # lat_xy = conv2dn(pad_lat_x, sy)
    # lat_yy = conv2dn(pad_lat_y, sy)
    # lat_grad = [latent, lat_x, lat_y, lat_xx, lat_xy, lat_yy]

    # w = [0, 1, 1, 2, 2, 2]

    # for i in range(15):
    #     for i in range(5):
    #         temp = conv2dn(lat_grad[i], psf) - img_grad[i]
    #         temp = np.linalg.norm(temp) ** 2
    #         temp *= 50 / (2 ** w[i])
    #     n2 = np.sum(np.abs(psf))

    m = psfModel(psf, img, latent)
    optim = torch.optim.Adam(m.parameters(), lr=0.001)
    training_loop(m, optim)
    for param in m.parameters():
        return param


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


def optimizeFF(img, latent, omega=None, psi=None, psf=None):
    pad_img = padding(img, 1)
    pad_lat = padding(latent, 1)
    sx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    img_x = conv2dn(pad_img, sx)
    img_y = conv2dn(pad_img, sy)
    pad_img_x = padding(img_x, 1)
    pad_img_y = padding(img_y, 1)
    img_xx = conv2dn(pad_img_x, sx)
    img_xy = conv2dn(pad_img_x, sy)
    img_yy = conv2dn(pad_img_y, sy)
    img_grad = [img, img_x, img_y, img_xx, img_xy, img_yy]

    lat_x = conv2dn(pad_lat, sx)
    lat_y = conv2dn(pad_lat, sy)
    pad_lat_x = padding(lat_x, 1)
    pad_lat_y = padding(lat_y, 1)
    lat_xx = conv2dn(pad_lat_x, sx)
    lat_xy = conv2dn(pad_lat_x, sy)
    lat_yy = conv2dn(pad_lat_y, sy)
    lat_grad = [padding(latent, psf.shape[0]//2), padding(lat_x, psf.shape[0]//2), padding(lat_y, psf.shape[0]//2),
                padding(lat_xx, psf.shape[0]//2), padding(lat_xy, psf.shape[0]//2), padding(lat_yy, psf.shape[0]//2)]

    w = [0, 1, 1, 2, 2, 2]

    psf = psf.reshape(-1)
    ans = minn(solveF, psf, args=[img_grad, lat_grad, w])
    print(ans.x)


def eq(x):
    if abs(x) > 10:
        return -(6.1e-4 * (x**2) + 5.0)
    else:
        return -2.7 * abs(x)


def computeGlobalPrior(latent):
    for i in range(len(latent)-1):
        for j in range(len(latent[i])-1):
            print(1)
