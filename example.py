import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
# import wdsr_b
from option2 import parser
from wdsr_b import *
from args import *
from vivo import vivo

with torch.cuda.device(7):
	args = get_args()
	# net = WDSR_B(args)
	net = vivo.Vivo32ch2RBs()
	# net = models.densenet161()
	flops, params = get_model_complexity_info(net, (1, 960, 540), as_strings=True, print_per_layer_stat=True)
	# flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
	print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
	print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# 144P(256×144) 240p(426×240) 360P(640×360) 480P(854×480)