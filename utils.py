import numpy as np
import torch
import os
from cv2 import imwrite
import cv2
from stdf_laplacian import MFVQE
from skimage.metrics import structural_similarity as ssim
#from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision import utils
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import gridspec

def psnr(batch1, batch2, max_value=1):
    mse = torch.mean((batch1 - batch2)**2)
    return 20 * torch.log10(max_value/torch.sqrt(mse + 1e-10))


def delta_psnr(restored, compressed, gt):
    return psnr(restored, gt) - psnr(compressed, gt)


def add_batch_in_tensorboard(name, writer, data, it): # data size is B x T x C x H x W
    grid = utils.make_grid(data)
    grid = torch.clamp(grid, 0, 1)
    writer.add_image(name, grid, it)


def ssim_batch(batch1, batch2):
    B, C, _, _ = batch1.shape
    batch_ssim = 0
    for batch in range(B):
        seq_ssim = 0
        for frame in range(C):
            frame1 = batch1[batch, frame, ...]  # 0 because it is excepted to work on Y channel of YUV color space
            frame2 = batch2[batch, frame, ...]
            seq_ssim += ssim(frame1.numpy(), frame2.numpy(), data_range=1)
        seq_ssim /= C
        batch_ssim += seq_ssim
    return batch_ssim / C


def save_data(data, path, y_only=False):
    if y_only:
        data = data[0, 0].cpu().numpy()
    else:
        data = data[0].cpu().permute(1, 2, 0).numpy()
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    data = np.uint8(data * 255).clip(0, 255)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imwrite(path, data)


def create_model(name, levels, std):
    if name == 'XS':
        model = MFVQE(std=std, levels=levels, lapfeat=24, stdffeat=24, lapnb=1, laptconv=False)
    elif name == 'S':
        model = MFVQE(std=std, levels=levels, lapfeat=32, stdffeat=24, lapnb=1, laptconv=False)
    elif name == 'M':
        model = MFVQE(std=std, levels=levels, lapfeat=48, stdffeat=32, lapnb=1, laptconv=True)
    elif name == 'L':
        model = MFVQE(std=std, levels=levels, lapfeat=64, stdffeat=32, lapnb=1, laptconv=True)
    elif name == 'XL':
        model = MFVQE(std=std, levels=levels, lapfeat=64, stdffeat=32, lapnb=2, laptconv=True)
    else:
        raise NotImplementedError(f'No implementation for model {name} is found')
    return model


def show_laplacian(compressed, restored, gt):
    levels = len(compressed)
    gs = gridspec.GridSpec(levels, 3)
    for level in range(levels - 1, -1, -1):
        ax = plt.subplot(gs[level, 0])
        ax.imshow(compressed[level][0, 0].cpu(), vmin=0, vmax=1)
        ax = plt.subplot(gs[level, 1])
        ax.imshow(restored[level][0, 0].cpu(), vmin=0, vmax=1)
        ax = plt.subplot(gs[level, 2])
        ax.imshow(gt[level][0, 0].cpu(), vmin=0, vmax=1)
    plt.show()


def visualize_features(feat, title):
    dim = int(feat.shape[1] ** 0.5)
    feat = feat.cpu()
    gs = gridspec.GridSpec(dim, dim)
    for i in range(dim):
        for j in range(dim):
            ax = plt.subplot(gs[i, j])
            ax.imshow(feat[0, i * dim + j])
    plt.title(title)
    plt.show()
