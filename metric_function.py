from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
import cv2
from scipy.signal import convolve2d

import numpy as np


def ssim_metric(output, target):
    batch_size, seq_len, _, height, width = output.shape
    ssim_values_sum = np.zeros(batch_size)
    ssim_values_first = np.zeros(batch_size)

    for i in range(batch_size):
        output_img = output[i, 0, 0, :, :].detach().cpu().numpy() * 255
        target_img = target[i, 0, 0, :, :].detach().cpu().numpy() * 255
        ssim_values_first[i] = ssim(target_img, output_img, data_range=255)

        ssim_seq_sum = 0
        for j in range(seq_len):
            output_img = output[i, j, 0, :, :].detach().cpu().numpy() * 255
            target_img = target[i, j, 0, :, :].detach().cpu().numpy() * 255
            ssim_seq_sum += ssim(target_img, output_img, data_range=255)
        ssim_values_sum[i] = ssim_seq_sum / seq_len

    return np.mean(ssim_values_sum), np.mean(ssim_values_first)


def mse_metric(output, target):
    batch_size, seq_len, _, height, width = output.shape
    mse_values_sum = np.zeros(batch_size)
    mse_values_first = np.zeros(batch_size)

    for i in range(batch_size):
        output_img = output[i, 0, :, :, :].detach().cpu().numpy().flatten() * 255
        target_img = target[i, 0, :, :, :].detach().cpu().numpy().flatten() * 255
        mse_values_first[i] = mean_squared_error(target_img, output_img)

        mse_seq_sum = 0
        for j in range(seq_len):
            output_img = output[i, j, :, :, :].detach().cpu().numpy().flatten() * 255
            target_img = target[i, j, :, :, :].detach().cpu().numpy().flatten() * 255
            mse_seq_sum += mean_squared_error(target_img, output_img)
        mse_values_sum[i] = mse_seq_sum / seq_len

    return np.mean(mse_values_sum), np.mean(mse_values_first)

def psnr_metric(output, target):
    batch_size, seq_len, _, height, width = output.shape
    psnr_values_sum = np.zeros(batch_size)
    psnr_values_first = np.zeros(batch_size)

    for i in range(batch_size):
        output_img = output[i, 0, :, :, :].detach().cpu().numpy() * 255
        target_img = target[i, 0, :, :, :].detach().cpu().numpy() * 255
        psnr_values_first[i] = cv2.PSNR(target_img, output_img)

        psnr_seq_sum = 0
        for j in range(seq_len):
            output_img = output[i, j, :, :, :].detach().cpu().numpy() * 255
            target_img = target[i, j, :, :, :].detach().cpu().numpy() * 255
            psnr_seq_sum += cv2.PSNR(target_img, output_img)
        psnr_values_sum[i] = psnr_seq_sum / seq_len

    return np.mean(psnr_values_sum), np.mean(psnr_values_first)



# the method is from the papaer "No-Reference Image Sharpness Assessment Based on Maximum Gradient and Variability of Gradients"
def sharpness_calculate(img1):
    # define the filter
    F1 = np.array([[0, 0], [-1, 1]])
    F2 = F1.T

    # calculate the gradient of the image
    H1 = convolve2d(img1, F1, mode='valid')
    H2 = convolve2d(img1, F2, mode='valid')
    g = np.sqrt(H1 ** 2 + H2 ** 2)

    row, col = g.shape
    B = round(min(row, col) / 16)
    g_center = g[B + 1: -B, B + 1: -B]
    MaxG = np.max(g_center)
    MinG = np.min(g_center)
    CVG = (MaxG - MinG) / np.mean(g_center)
    re = MaxG ** 0.61 * CVG ** 0.39

    return re

def sharpness_metric(output):
    batch_size, seq_len, _, height, width = output.shape
    sharpness_values_sum = np.zeros(batch_size)
    sharpness_values_first = np.zeros(batch_size)

    for i in range(batch_size):
        output_img = output[i, 0, 0, :, :].detach().cpu().numpy() * 255
        sharpness_values_first[i] = sharpness_calculate(output_img)

        sharpness_seq_sum = 0
        for j in range(seq_len):
            output_img = output[i, j, 0, :, :].detach().cpu().numpy() * 255
            sharpness_seq_sum += sharpness_calculate(output_img)
        sharpness_values_sum[i] = sharpness_seq_sum / seq_len

    return np.mean(sharpness_values_sum), np.mean(sharpness_values_first)