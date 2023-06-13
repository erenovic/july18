import torch
import torch.nn as nn
# import torchsparse
# import torchsparse.nn as spnn
# from torchsparse.tensor import PointTensor

# from tsparse.torchsparse_utils import *

from .sparseconvnet.convolution import Convolution
from .sparseconvnet.deconvolution import Deconvolution

from .sparseconvnet.batchNormalization import BatchNormalization
from .sparseconvnet.activations import ReLU


# __all__ = ['SPVCNN', 'SConv3d', 'SparseConvGRU']


class BasicSparseConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, dimension=3):
        super().__init__()
        self.net = nn.Sequential(
            Convolution(
                dimension, nIn=inc, nOut=outc, filter_size=ks,
                dilation=dilation, filter_stride=1, stride=stride
            ),
            BatchNormalization(outc),
            ReLU()
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicSparseDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dimension=3):
        super().__init__()
        self.net = nn.Sequential(
            Deconvolution(
                dimension=dimension, nIn=inc, nOut=outc, 
                filter_size=ks, filter_stride=stride, 
                bias=True
            ),
            BatchNormalization(outc),
            ReLU()
        )

    def forward(self, x):
        return self.net(x)


class SparseCostRegNet(nn.Module):
    """
    Sparse cost regularization network;
    require sparse tensors as input
    """

    def __init__(self, d_in, d_out=8):
        super(SparseCostRegNet, self).__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.conv0 = BasicSparseConvolutionBlock(d_in, d_out)

        self.conv1 = BasicSparseConvolutionBlock(d_out, 16, stride=2)
        self.conv2 = BasicSparseConvolutionBlock(16, 16)

        self.conv3 = BasicSparseConvolutionBlock(16, 32, stride=2)
        self.conv4 = BasicSparseConvolutionBlock(32, 32)

        self.conv5 = BasicSparseConvolutionBlock(32, 64, stride=2)
        self.conv6 = BasicSparseConvolutionBlock(64, 64)

        self.conv7 = BasicSparseDeconvolutionBlock(64, 32, ks=3, stride=2)

        self.conv9 = BasicSparseDeconvolutionBlock(32, 16, ks=3, stride=2)

        self.conv11 = BasicSparseDeconvolutionBlock(16, d_out, ks=3, stride=2)

        self.input_layer = 

    def forward(self, x):
        """

        :param x: sparse tensor
        :return: sparse tensor
        """
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))

        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        return x.F