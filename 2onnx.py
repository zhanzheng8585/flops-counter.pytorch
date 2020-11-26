import os, time, shutil, argparse
from functools import partial
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from collections import OrderedDict
import torch.utils.data
import torch.utils.data.distributed
import torch.onnx as torch_onnx
import onnx

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from skimage import io

import functools
import torch.nn.functional as F


# def make_layer(block, n_layers):
#     layers = []
#     for _ in range(n_layers):
#         layers.append(block())
#     return nn.Sequential(*layers)


# class ResidualBlock_noBN(nn.Module):
#     '''Residual block w/o BN
#     ---Conv-ReLU-Conv-+-
#      |________________|
#     '''

#     def __init__(self, nf=64):
#         super(ResidualBlock_noBN, self).__init__()
#         self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

#     def forward(self, x):
#         identity = x
#         out = F.relu(self.conv1(x), inplace=True)
#         out = self.conv2(out)
#         return identity + out


# class Vivo32ch2RBs(nn.Module):
#     def __init__(self, nf=32, nRBs=2):
#         super().__init__()
#         self.relu = nn.ReLU()
#         self.conv_in = nn.Conv2d(1, nf, 3, 1, 1)
#         RB_noBn = functools.partial(ResidualBlock_noBN, nf=nf)
#         self.sr = make_layer(RB_noBn, nRBs)
#         self.conv_out = nn.Conv2d(nf, 4, 3, 1, 1)
#         self.pix_shuffle = nn.PixelShuffle(2)
    
#     def forward(self, x):
#         y = self.relu(self.conv_in(x))
#         y = self.sr(y)
#         y = self.pix_shuffle(self.conv_out(y))

#         return y


# class Vivo16ch7RBs(nn.Module):
#     def __init__(self, nf=16, nRBs=7):
#         super().__init__()
#         self.relu = nn.ReLU()
#         self.conv_in = nn.Conv2d(1, nf, 3, 1, 1)
#         RB_noBn = functools.partial(ResidualBlock_noBN, nf=nf)
#         self.sr = make_layer(RB_noBn, nRBs)
#         self.conv_out = nn.Conv2d(nf, 4, 3, 1, 1)
#         self.pix_shuffle = nn.PixelShuffle(2)
    
#     def forward(self, x):
#         y = self.relu(self.conv_in(x))
#         y = self.sr(y)
#         y = self.pix_shuffle(self.conv_out(y))

#         return y


# class Vivo8ch29RBs(nn.Module):
#     def __init__(self, nf=16, nRBs=7):
#         super().__init__()
#         self.relu = nn.ReLU()
#         self.conv_in = nn.Conv2d(1, nf, 3, 1, 1)
#         RB_noBn = functools.partial(ResidualBlock_noBN, nf=nf)
#         self.sr = make_layer(RB_noBn, nRBs)
#         self.conv_out = nn.Conv2d(nf, 4, 3, 1, 1)
#         self.pix_shuffle = nn.PixelShuffle(2)
    
#     def forward(self, x):
#         y = self.relu(self.conv_in(x))
#         y = self.sr(y)
#         y = self.pix_shuffle(self.conv_out(y))

#         return y

# import prune_util
# from prune_util import GradualWarmupScheduler
# from prune_util import CrossEntropyLossMaybeSmooth
# from prune_util import mixup_data, mixup_criterion

# from utils import save_checkpoint, AverageMeter, visualize_image, GrayscaleImageFolder
# from model import ColorNet
from wdsr_b import *
from args import *
from models import *
from vivo import vivo
import argparse
import importlib

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',help='Dataset name.',default="div2k",type=str)
parser.add_argument('--model',help='Model name.',default='scn',type=str)
parser.add_argument('--job_dir',help='Directory to write checkpoints and export models.')
parser.add_argument('--ckpt',help='File path to load checkpoint.',default=None,type=str,)
parser.add_argument('--eval_only',default=False,action='store_true',help='Running evaluation only.',)
parser.add_argument('--eval_datasets',help='Dataset names for evaluation.',default=None,type=str,nargs='+',)
# Experiment arguments
parser.add_argument('--save_checkpoints_epochs',help='Number of epochs to save checkpoint.',default=1,type=int)
parser.add_argument('--train_epochs',help='Number of epochs to run training totally.',default=10,type=int)
parser.add_argument('--log_steps',help='Number of steps for training logging.',default=100,type=int)
parser.add_argument('--random_seed',help='Random seed for TensorFlow.',default=None,type=int)
# Performance tuning parameters
parser.add_argument('--opt_level',help='Number of GPUs for experiments.',default='O0',type=str)
parser.add_argument('--sync_bn',default=False,action='store_true',help='Enabling apex sync BN.')
# Verbose
parser.add_argument('-v','--verbose',action='count',default=0,help='Increasing output verbosity.',)
parser.add_argument('--scale', default=2, type=int)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--node_rank', default=0, type=int)

# Parse arguments
args, _ = parser.parse_known_args()

model_module = importlib.import_module('models.' +
                                     args.model if args.model else 'models')
model_module.update_argparser(parser)
params = parser.parse_args()
model, criterion, optimizer, lr_scheduler, metrics = model_module.get_model_spec(params)


def main():

    use_gpu = torch.cuda.is_available()
    # Create model  
    # models.resnet18(num_classes=365)
    # model = ColorNet()
    # args = get_args()
    # model = MODEL(args)
    model = vivo.Vivo32ch2RBs()
    # state_dict = torch.load("./checkpoint/checkpoint6/model_epoch133_step1.pth")
    # new_state_dict = OrderedDict()

    # for k, v in state_dict.items():
    #     k = k.replace('module.', '')
    #     new_state_dict[k] = v

    # model = torch.nn.DataParallel(model)
    # model.load_state_dict(new_state_dict)
    checkpoint = torch.load("/home/zhanzheng/flops-counter.pytorch/vivo/vivo_32ch_2rbs.pth")
    model.load_state_dict(new_state_dict)
    # model.load_state_dict(checkpoint["model"].state_dict())

    # print(model)
    input_shape = (1, 960, 540)
    model_onnx_path = "./Vivo32ch2RBs.onnx"
    model.train(False)
    model.eval()

    # Export the model to an ONNX file
    dummy_input = Variable(torch.randn(1, *input_shape))
    output = torch_onnx.export(model, 
                              dummy_input, 
                              model_onnx_path, 
                              verbose=False)
    print("Export of torch_model.onnx complete!")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def check():

    model = vivo.Vivo32ch2RBs()
    checkpoint = torch.load("/home/zhanzheng/flops-counter.pytorch/vivo/vivo_32ch_2rbs.pth")
    model.load_state_dict(new_state_dict)
    # model.load_state_dict(checkpoint["model"].state_dict())

    # torch.nn.utils.remove_weight_norm(model.head[0])
    # for i in range(1):
    #     for j in [0,2,3]:
    #         torch.nn.utils.remove_weight_norm(model.body[i].body[j])
    # torch.nn.utils.remove_weight_norm(model.tail[0])
    # torch.nn.utils.remove_weight_norm(model.skip[0])

    model.eval()
    ort_session = onnxruntime.InferenceSession("Vivo32ch2RBs.onnx")

    x = torch.randn(1, 1, 960, 540, requires_grad=False)
    torch_out = model(x)
    # # Load the ONNX model
    # model = onnx.load("wdsr_b.onnx")

    # # Check that the IR is well formed
    # onnx.checker.check_model(model)

    # # Print a human readable representation of the graph
    # onnx.helper.printable_graph(model.graph)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == '__main__':
    main()
    check()
