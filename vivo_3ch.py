import torch
import torch.nn as nn
import functools
import torch.nn.functional as F

def make_model(args, parent=False):
    return Vivo3ch(args)

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


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)

# def de_subpix(y):
#     (b, c, h, w) = y.shape
#     # print(b, c, h, w)
#     h1 = int(h / 2)
#     w1 = int(w / 2)
#     d1 = torch.zeros((b, c, h1, w1))
#     d2 = torch.zeros((b, c, h1, w1))
#     d3 = torch.zeros((b, c, h1, w1))
#     d4 = torch.zeros((b, c, h1, w1))
#     # print(y.shape)
#     for i in range(0, h1, 2):
#         for j in range(0, w1, 2):
#             d1[:, :, i, j] = y[:, :, 2 * i, 2 * j]
#             d2[:, :, i, j] = y[:, :, 2 * i + 1, 2 * j]
#             d3[:, :, i, j] = y[:, :, 2 * i, 2 * j + 1]
#             d4[:, :, i, j] = y[:, :, 2 * i + 1, 2 * j + 1]
#             # print()
#             # print(i,j)
#     out = torch.cat([d1, d2, d3, d4], 1).cuda()
#     # print(out.shape)
#     return out

# class DeSubpixelConv2d(nn.Module):
#     def __init__(self, prev_layer, scale=2, act=None, name='desubpixel_conv2d'):
#         super().__init__()

#         # Assume the input have desired shape (width and height are divided by scale)
#         self.outputs = self._apply_activation(self._PDS(self.inputs, r=scale))
        
#         self._add_layers(self.outputs)

#     def _PDS(self, X, r):
#         X = space_to_depth(X, r)
#         return X


# class Vivo3ch(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         scale = args.scale
#         nf = args.n_feats
#         nRBs = args.n_resblocks
#         self.relu = nn.ReLU()
#         self.conv_in = nn.Conv2d(3, nf//4, 1, (1, 1), padding=1//2)
#         RB_noBn = functools.partial(ResidualBlock_noBN, nf=nf)
#         self.sr = make_layer(RB_noBn, nRBs)
#         self.conv_out1 = nn.Conv2d(nf, nf*4, 1, (1, 1), padding=1//2)
#         self.conv_out2 = nn.Conv2d(nf, 3*4, 1, (1, 1), padding=1//2)
#         self.pix_shuffle = nn.PixelShuffle(scale)
    
#     def forward(self, x):
#         # print(x.size())
#         y = self.relu(space_to_depth(self.conv_in(x), 2))
#         # print(y.size())
#         y = self.sr(y)
#         # print(y.size())
#         y = self.pix_shuffle(self.conv_out1(y))
#         y = self.pix_shuffle(self.conv_out2(y))
#         # print(y.size())

#         return y


class Vivo3ch(nn.Module):
    def __init__(self, args):
        super().__init__()
        scale = args.scale
        nf = args.n_feats
        nRBs = args.n_resblocks
        self.relu = nn.ReLU()
        self.conv_in = nn.Conv2d(3, nf, 3, 1, 1)
        RB_noBn = functools.partial(ResidualBlock_noBN, nf=nf)
        self.sr = make_layer(RB_noBn, nRBs)
        self.conv_out = nn.Conv2d(nf, 4*3, 3, 1, 1)
        self.pix_shuffle = nn.PixelShuffle(scale)
    
    def forward(self, x):
        y = self.relu(self.conv_in(x))
        y = self.sr(y)
        y = self.pix_shuffle(self.conv_out(y))

        return y


class Vivo32ch2RBs3ch(nn.Module):
    def __init__(self, nf=32, nRBs=2):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv_in = nn.Conv2d(3, nf, 3, 1, 1)
        RB_noBn = functools.partial(ResidualBlock_noBN, nf=nf)
        self.sr = make_layer(RB_noBn, nRBs)
        self.conv_out = nn.Conv2d(nf, 4*3, 3, 1, 1)
        self.pix_shuffle = nn.PixelShuffle(2)
    
    def forward(self, x):
        y = self.relu(self.conv_in(x))
        y = self.sr(y)
        y = self.pix_shuffle(self.conv_out(y))

        return y


class Vivo16ch7RBs3ch(nn.Module):
    def __init__(self, nf=16, nRBs=7):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv_in = nn.Conv2d(3, nf, 3, 1, 1)
        RB_noBn = functools.partial(ResidualBlock_noBN, nf=nf)
        self.sr = make_layer(RB_noBn, nRBs)
        self.conv_out = nn.Conv2d(nf, 4*3, 3, 1, 1)
        self.pix_shuffle = nn.PixelShuffle(2)
    
    def forward(self, x):
        y = self.relu(self.conv_in(x))
        y = self.sr(y)
        y = self.pix_shuffle(self.conv_out(y))

        return y


class Vivo8ch29RBs3ch(nn.Module):
    def __init__(self, nf=8, nRBs=29):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv_in = nn.Conv2d(3, nf, 3, 1, 1)
        RB_noBn = functools.partial(ResidualBlock_noBN, nf=nf)
        self.sr = make_layer(RB_noBn, nRBs)
        self.conv_out = nn.Conv2d(nf, 4*3, 3, 1, 1)
        self.pix_shuffle = nn.PixelShuffle(2)
    
    def forward(self, x):
        y = self.relu(self.conv_in(x))
        y = self.sr(y)
        y = self.pix_shuffle(self.conv_out(y))

        return y
