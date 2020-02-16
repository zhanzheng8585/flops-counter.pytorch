import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
import wdsr
from option2 import parser

args = parser.parse_args()
with torch.cuda.device(6):
	net = wdsr.WDSR_B(args)
	# net = models.densenet161()
	flops, params = get_model_complexity_info(net, (3, 256, 256), as_strings=True, print_per_layer_stat=True)
	# flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
	print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
	print('{:<30}  {:<8}'.format('Number of parameters: ', params))

