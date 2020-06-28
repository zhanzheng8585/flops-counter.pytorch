import torch
import torch.nn as nn
import functools
import torch.nn.functional as F


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class Vivo32ch2RBs(nn.Module):
    def __init__(self, nf=32, nRBs=2):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv_in = nn.Conv2d(1, nf, 3, 1, 1)
        RB_noBn = functools.partial(ResidualBlock_noBN, nf=nf)
        self.sr = make_layer(RB_noBn, nRBs)
        self.conv_out = nn.Conv2d(nf, 4, 3, 1, 1)
        self.pix_shuffle = nn.PixelShuffle(2)
    
    def forward(self, x):
        y = self.relu(self.conv_in(x))
        y = self.sr(y)
        y = self.pix_shuffle(self.conv_out(y))

        return y


class Vivo16ch7RBs(nn.Module):
    def __init__(self, nf=16, nRBs=7):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv_in = nn.Conv2d(1, nf, 3, 1, 1)
        RB_noBn = functools.partial(ResidualBlock_noBN, nf=nf)
        self.sr = make_layer(RB_noBn, nRBs)
        self.conv_out = nn.Conv2d(nf, 4, 3, 1, 1)
        self.pix_shuffle = nn.PixelShuffle(2)
    
    def forward(self, x):
        y = self.relu(self.conv_in(x))
        y = self.sr(y)
        y = self.pix_shuffle(self.conv_out(y))

        return y


class Vivo8ch29RBs(nn.Module):
    def __init__(self, nf=8, nRBs=29):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv_in = nn.Conv2d(1, nf, 3, 1, 1)
        RB_noBn = functools.partial(ResidualBlock_noBN, nf=nf)
        self.sr = make_layer(RB_noBn, nRBs)
        self.conv_out = nn.Conv2d(nf, 4, 3, 1, 1)
        self.pix_shuffle = nn.PixelShuffle(2)
    
    def forward(self, x):
        y = self.relu(self.conv_in(x))
        y = self.sr(y)
        y = self.pix_shuffle(self.conv_out(y))

        return y