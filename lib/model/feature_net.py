

import torch
import torch.nn as nn
import torch.nn.functional as F


def ConvBnReLU(ch_in, ch_out, kernel, stride, padding):
    return nn.Sequential(
        nn.Conv2d(
            ch_in, ch_out, kernel, stride, padding
        ),
        nn.BatchNorm2d(ch_out),
        nn.ReLU()
    )

class FeatureNet(nn.Module):
    """
    output 3 levels of features using a FPN structure
    """

    def __init__(self):
        super(FeatureNet, self).__init__()

        self.conv0 = nn.Sequential(
            ConvBnReLU(3, 8, 3, 1, 1),
            ConvBnReLU(8, 8, 3, 1, 1))

        self.conv1 = nn.Sequential(
            ConvBnReLU(8, 16, 5, 2, 2),
            ConvBnReLU(16, 16, 3, 1, 1))

        self.conv2 = nn.Sequential(
            ConvBnReLU(16, 32, 5, 2, 2),
            ConvBnReLU(32, 32, 3, 1, 1))

        self.toplayer = nn.Conv2d(32, 32, 1)
        self.lat1 = nn.Conv2d(16, 32, 1)
        self.lat0 = nn.Conv2d(8, 32, 1)

        # to reduce channel size of the outputs from FPN
        self.smooth1 = nn.Conv2d(32, 16, 3, padding=1)
        self.smooth0 = nn.Conv2d(32, 8, 3, padding=1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2,
                             mode="bilinear", align_corners=True) + y

    def forward(self, x):
        # x: (B, 3, H, W)
        conv0 = self.conv0(x)  # (B, 8, H, W)
        conv1 = self.conv1(conv0)  # (B, 16, H//2, W//2)
        conv2 = self.conv2(conv1)  # (B, 32, H//4, W//4)
        feat2 = self.toplayer(conv2)  # (B, 32, H//4, W//4)
        feat1 = self._upsample_add(feat2, self.lat1(conv1))  # (B, 32, H//2, W//2)
        feat0 = self._upsample_add(feat1, self.lat0(conv0))  # (B, 32, H, W)

        # reduce output channels
        feat1 = self.smooth1(feat1)  # (B, 16, H//2, W//2)
        feat0 = self.smooth0(feat0)  # (B, 8, H, W)

        # feats = {"level_0": feat0,
        #          "level_1": feat1,
        #          "level_2": feat2}

        return [feat2, feat1, feat0]  # coarser to finer features