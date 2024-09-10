import torch
import torch.nn as nn
import torch.nn.functional as F
from ops.dcn.deform_conv import ModulatedDeformConv
import numpy as np
import cv2
import utils

# ==========
# Spatio-temporal deformable fusion module
# ==========

class STDF(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, base_ks=3, deform_ks=3):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super(STDF, self).__init__()

        self.nb = nb
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2

        # u-shape backbone
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )
        for i in range(1, nb):
            setattr(
                self, 'dn_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True)
                )
            )
            setattr(
                self, 'up_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(2 * nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )

        # regression head
        # why in_nc*3*size_dk?
        #   in_nc: each map use individual offset and mask
        #   2*size_dk: 2 coordinates for each point
        #   1*size_dk: 1 confidence (attention) score for each point
        self.offset_mask = nn.Conv2d(
            nf, in_nc * 3 * self.size_dk, base_ks, padding=base_ks // 2
        )

        # deformable conv
        # notice group=in_nc, i.e., each map use individual offset and mask
        self.deform_conv = ModulatedDeformConv(
            in_nc, out_nc, deform_ks, padding=deform_ks // 2, deformable_groups=in_nc
        )

    def forward(self, inputs):
        nb = self.nb
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks

        # feature extraction (with downsampling)
        out_lst = [self.in_conv(inputs)]  # record feature maps for skip connections
        for i in range(1, nb):
            dn_conv = getattr(self, 'dn_conv{}'.format(i))
            out_lst.append(dn_conv(out_lst[i - 1]))
        # trivial conv
        out = self.tr_conv(out_lst[-1])
        # feature reconstruction (with upsampling)
        for i in range(nb - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out = up_conv(
                torch.cat([out, out_lst[i]], 1)
            )

        # compute offset and mask
        # offset: conv offset
        # mask: confidence
        off_msk = self.offset_mask(self.out_conv(out))
        off = off_msk[:, :in_nc * 2 * n_off_msk, ...]
        msk = torch.sigmoid(
            off_msk[:, in_nc * 2 * n_off_msk:, ...]
        )

        # perform deformable convolutional fusion
        fused_feat = F.relu(
            self.deform_conv(inputs, off, msk),
            inplace=True
        )

        return fused_feat


# ==========
# Quality enhancement module
# ==========


class LaplacianNet(nn.Module):  # each level has its own network, but details are treated using the same network

    class Encoder(nn.Module):

        def __init__(self, in_ch, out_ch, stride=1, nb=1):
            super(LaplacianNet.Encoder, self).__init__()
            self.net = nn.Sequential()
            self.net.add_module('input_conv', nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride))
            for i in range(nb - 1):
                self.net.add_module(f'mid_relu_{i}', nn.ReLU(inplace=True))
                self.net.add_module(f'mid_conv_{i}', nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
            self.net.add_module('out_relu', nn.ReLU(inplace=True))
            self.net.add_module('out_conv', nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        def forward(self, x):
            return self.net(x)

    class Feat2Im(nn.Module):

        def __init__(self, in_ch, out_ch):
            super(LaplacianNet.Feat2Im, self).__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch // 2, out_ch, kernel_size=3, padding=1)
            )

        def forward(self, x):
            return self.net(x)

    class Decoder(nn.Module):

        def __init__(self, ch, nb=1, use_tconv=False):
            super(LaplacianNet.Decoder, self).__init__()
            self.up = None
            if use_tconv:
                self.up = nn.Sequential(nn.ConvTranspose2d(ch, ch, kernel_size=4, padding=1, stride=2),
                                        nn.ReLU(inplace=True)
                                        )
            self.net = nn.Sequential()
            self.net.add_module('input_conv', nn.Conv2d(ch, ch, kernel_size=3, padding=1))
            for i in range(nb - 1):
                self.net.add_module(f'mid_relu_{i}', nn.ReLU(inplace=True))
                self.net.add_module(f'mid_conv_{i}', nn.Conv2d(ch, ch, kernel_size=3, padding=1))
            self.net.add_module('out_relu', nn.ReLU(inplace=True))
            self.net.add_module('out_conv', nn.Conv2d(ch, ch, kernel_size=3, padding=1))

        def forward(self, x, y):
            if self.up is not None: # it depends whether the upscaling is performed using bicubic or tconv
                return self.net(x + self.up(y))
            else:
                target_size = (x.shape[2], x.shape[3])
                return self.net(x + nn.functional.interpolate(y, size=target_size, mode='bicubic', align_corners=False))

    def __init__(self, in_ch=32, out_ch=1, mid_ch=32, nb=1, levels=5, tconv=True):
        super(LaplacianNet, self).__init__()
        self.levels = levels
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.feat_2_im = nn.ModuleList()

        for level in range(levels):
            if level == 0:
                e = LaplacianNet.Encoder(in_ch, mid_ch, nb=nb)
            else:
                e = LaplacianNet.Encoder(mid_ch, mid_ch, stride=2, nb=nb)
            self.encoders.extend(nn.Sequential(e))

        for _ in range(levels):
            self.feat_2_im.extend(nn.Sequential(LaplacianNet.Feat2Im(mid_ch, out_ch)))

        for _ in range(levels - 1):
            self.decoders.extend(nn.Sequential(LaplacianNet.Decoder(mid_ch, nb=nb, use_tconv=tconv)))

    def forward(self, x):
    
        encoded = [self.encoders[0](x)] # initial encoding
        for level in range(1, self.levels):
            encoder = self.encoders[level]
            prev_encoded = encoded[level - 1]
            curr_encoded = encoder(prev_encoded)
            encoded.append(curr_encoded)

        decoded = [encoded[len(encoded) - 1]] # last encoded layer
        for level in range(1, self.levels):
            decoder = self.decoders[level - 1]
            prev_decoded = decoded[level - 1]
            curr_encoded = encoded[len(encoded) - 1 - level]
            curr_decoded = decoder(curr_encoded, prev_decoded)
            decoded.append(curr_decoded)

        laps = []
        for level in range(len(self.feat_2_im)):
            curr_decoded = decoded[level]
            curr_lap = self.feat_2_im[level](curr_decoded)
            laps.append(curr_lap)

        return laps


# ==========
# MFVQE network
# ==========

class MFVQE(nn.Module):
    """STDF -> QE -> residual.

    in: (B T C H W)
    out: (B C H W)
    """

    def __init__(self, std=1, levels=5, lapfeat=32, stdffeat=24, lapnb=1, laptconv=True):
        """
        Arg:
            opts_dict: network parameters defined in YAML.
        """
        super(MFVQE, self).__init__()

        self.radius = 3
        self.input_len = 2 * self.radius + 1
        self.in_nc = 1
        self.ffnet = STDF(
            in_nc=self.in_nc * self.input_len,
            out_nc=lapfeat,
            nf=stdffeat,
            nb=3,
            deform_ks=3
        )
        self.levels = levels
        self.lap = LaplacianLossImages.Laplacian(levels=levels, std=std)
        self.lapnet = LaplacianNet(in_ch=lapfeat, mid_ch=lapfeat, nb=lapnb, levels=levels, tconv=laptconv)

    def forward(self, x, gt_y=None, map=None,):
        central_frame = x[:, self.radius:self.radius+1]
        central_lap_pyr = self.lap.laplacian_pyramid(central_frame)
        aligned_feat = self.ffnet(x)
        out = self.lapnet(aligned_feat)
        int_result = central_lap_pyr[0] + out[0]
        laps = [int_result]
        result = [int_result]
        for i in range(1, self.levels):
            lap = central_lap_pyr[i] + out[i]
            laps.append(lap)
            target_size = (lap.shape[2], lap.shape[3])
            int_result = nn.functional.interpolate(int_result, size=target_size, mode='bicubic', align_corners=False) + lap
            result.append(int_result)
        return result[len(result) - 1], result, laps


class LaplacianLossImages(nn.Module):
    class Laplacian(nn.Module):

        @staticmethod
        def gkern(kernlen, std):
            kernel = cv2.getGaussianKernel(kernlen, std).dot(cv2.getGaussianKernel(kernlen, std).transpose())
            return nn.Parameter(torch.from_numpy(kernel.astype(np.float32)).unsqueeze(0).unsqueeze(0), requires_grad=False)

        def __init__(self, std=1, levels=3):
            super(LaplacianLossImages.Laplacian, self).__init__()
            self.levels = levels
            self.ksize = int(1 + 2 * np.floor(std))
            self.gaussian_kernel = LaplacianLossImages.Laplacian.gkern(self.ksize, std)
            self.pad = (self.ksize // 2, self.ksize // 2, self.ksize // 2, self.ksize // 2)

        def laplacian_pyramid(self, im):
            gauss_pyramid = self.gaussian_pyramid(im)
            lap_pyramid = [gauss_pyramid[0]]
            for level in range(self.levels - 1):
                im_gauss = gauss_pyramid[level]
                target_size = (gauss_pyramid[level + 1].shape[2], gauss_pyramid[level + 1].shape[3])
                im_gauss = nn.functional.interpolate(im_gauss, size=target_size, mode='bicubic', align_corners=False)
                lap_pyramid.append(gauss_pyramid[level + 1] - im_gauss)
            return lap_pyramid

        def gaussian_pyramid(self, im):
            gauss_pyramid = [im]
            for level in range(self.levels - 1):
                im = nn.functional.conv2d(nn.functional.pad(im, self.pad, mode='reflect'), self.gaussian_kernel)
                target_size = (int(np.ceil(im.shape[2] / 2)), int(np.ceil(im.shape[3] / 2)))
                im = nn.functional.interpolate(im, size=target_size, mode='bicubic', align_corners=False)
                gauss_pyramid.append(im)
            gauss_pyramid.reverse()
            return gauss_pyramid

        def compose_levels(self, laplacian_pyramid):
            im = laplacian_pyramid[0]
            reconstruction = [im]
            for level in laplacian_pyramid[1:]:
                target_size = (level.shape[2], level.shape[3])
                im = nn.functional.interpolate(im, size=target_size, mode='bicubic') + level
                reconstruction.append(im)
            return reconstruction

    def __init__(self, std=1, levels=4, is_test=False, loss='l2'):
        super(LaplacianLossImages, self).__init__()
        self.laplacian = self.Laplacian(std, levels)
        self.is_test = is_test
        if loss == 'l2':
            self.loss = self.l2
        elif loss == 'l1':
            self.loss = self.l1
        else:
            raise NotImplementedError(f'No implementation for loss {loss}')

    def l2(self, x, y):
        return ((x - y) ** 2).mean()

    def l1(self, x, y):
        return (x - y).abs().mean()

    def forward(self, restored, gt, compressed=None, apply_to_images=True):
        gt = self.laplacian.laplacian_pyramid(gt)
        if apply_to_images:
            gt = self.laplacian.compose_levels(gt)
        if compressed is not None:
            compressed = self.laplacian.compose_levels(self.laplacian.laplacian_pyramid(compressed))
        loss = 0
        for level in range(len(gt)):
            tmp = self.loss(restored[level], gt[level])
            loss += tmp
        if self.is_test and compressed is not None:
            utils.show_laplacian(compressed, restored, gt)
        return loss


