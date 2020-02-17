import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
import wdsr_b
from option2 import parser
import args

with torch.cuda.device(6):
	args = get_args()
	net = wdsr_b.MODEL(args)
	# net = models.densenet161()
	flops, params = get_model_complexity_info(net, (3, 128, 128), as_strings=True, print_per_layer_stat=True)
	# flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
	print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
	print('{:<30}  {:<8}'.format('Number of parameters: ', params))

