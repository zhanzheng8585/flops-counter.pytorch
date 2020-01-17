import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
from model import ColorNet

with torch.cuda.device(6):
	net = model.ColorNet()
	# net = models.densenet161()
	flops, params = get_model_complexity_info(net, (1, 224, 224), as_strings=True, print_per_layer_stat=True)
	# flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
	print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
	print('{:<30}  {:<8}'.format('Number of parameters: ', params))

