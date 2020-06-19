from __future__ import print_function
import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
import argparse
import os
import sys
# from resnet_1d import ResNet50_1d
# from resnet_1d_lite import ResNet50_1d_shrink
from thop import profile
import yaml
import wdsr_b
from option2 import parser
from wdsr_b import *
from args import *

# parser = argparse.ArgumentParser(description='Load Models')
# parser.add_argument('--slice_size', type=int, default=198, help='input size')
# parser.add_argument('--devices', type=int, default=500, help='number of classes')


with torch.cuda.device(0):
	# args = parser.parse_args()
	# shrink = 0.547
	# base_path = os.getcwd()
	# model_folder = ['1C_wifi_raw', '1C_adsb', '1C_mixture','1C_wifi_eq','1A_wifi_eq']
	# slice_sizes = [512,512,512,198,198]
	# devices = [50,50,50,50,500]
	# for folder, slice_size, device in zip(model_folder,slice_sizes,devices):
	# 	print(folder)
	# model = ResNet50_1d(args.slice_size,args.devices)
	args = get_args()
	model = WDSR_B(args)
	input = torch.randn(3,320,180)

	model.train(False)
	model.eval()
	macs, params = profile(model, inputs=(input, ))
	# flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
	print('{:<30}  {:<8}'.format('Computational complexity: ', macs * 2/1000000000)) # GMACs
	print('{:<30}  {:<8}'.format('Number of parameters: ', params/1000000)) # M


	# model = ResNet50_1d_shrink(args.slice_size,args.devices,shrink)
	# input = torch.randn(1, 2, args.slice_size)

	# config = "./modelprofile/config_res50_v5.yaml"
	# if not isinstance(config, str):
	# 	raise Exception("filename must be a str")
	# with open(config, "r") as stream:
	# 	try:
	# 		raw_dict = yaml.load(stream)
	# 		prune_ratios = raw_dict['prune_ratios']
	# 		for k,v in prune_ratios.items():
	# 			prune_ratios[k] = 0

	# 		model.train(False)
	# 		model.eval()
	# 		macs, params = profile(model, inputs=(input, ), rate = prune_ratios)
	# 		# flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
	# 		print('{:<30}  {:<8}'.format('Computational complexity: ', macs * 2/1000000000)) # GMACs
	# 		print('{:<30}  {:<8}'.format('Number of parameters: ', params/1000000)) # M
	# 	except yaml.YAMLError as exc:
	# 		print(exc)

