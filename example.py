import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
# import wdsr_b
# import argparse
import importlib
# from option import parser
from wdsr_b import *
from args import *
from vivo_3ch import *
from wdsr_a import *
# from models import CNNCifar
# from models import MLP
# from models import CNNMnist

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset',help='Dataset name.',default="div2k",type=str)
# parser.add_argument('--model',help='Model name.',default='scn',type=str)
# parser.add_argument('--job_dir',help='Directory to write checkpoints and export models.')
# parser.add_argument('--ckpt',help='File path to load checkpoint.',default=None,type=str,)
# parser.add_argument('--eval_only',default=False,action='store_true',help='Running evaluation only.',)
# parser.add_argument('--eval_datasets',help='Dataset names for evaluation.',default=None,type=str,nargs='+',)
# # Experiment arguments
# parser.add_argument('--save_checkpoints_epochs',help='Number of epochs to save checkpoint.',default=1,type=int)
# parser.add_argument('--train_epochs',help='Number of epochs to run training totally.',default=10,type=int)
# parser.add_argument('--log_steps',help='Number of steps for training logging.',default=100,type=int)
# parser.add_argument('--random_seed',help='Random seed for TensorFlow.',default=None,type=int)
# # Performance tuning parameters
# parser.add_argument('--opt_level',help='Number of GPUs for experiments.',default='O0',type=str)
# parser.add_argument('--sync_bn',default=False,action='store_true',help='Enabling apex sync BN.')
# # Verbose
# parser.add_argument('-v','--verbose',action='count',default=0,help='Increasing output verbosity.',)
# parser.add_argument('--scale', default=2, type=int)
# parser.add_argument('--local_rank', default=0, type=int)
# parser.add_argument('--node_rank', default=0, type=int)

# # Parse arguments
# args, _ = parser.parse_known_args()

# model_module = importlib.import_module('models.' +
#                                      args.model if args.model else 'models')
# model_module.update_argparser(parser)
# params = parser.parse_args()
# model, criterion, optimizer, lr_scheduler, metrics = model_module.get_model_spec(params)

with torch.cuda.device(7):

	args = get_args()


	net = WDSR_A(args)
	# args.num_classes = 10
	# net = model
	# net = MLP(dim_in = 1024, dim_hidden = 200, dim_out = 10)
	# net = CNNMnist(args=args)
	# net = vivo.Vivo8ch29RBs()
	# net = models.densenet161()
	flops, params = get_model_complexity_info(net, (3, 128, 128), as_strings=True, print_per_layer_stat=True)
	# flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
	print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
	print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# 144P(256×144) 240p(426×240) 360P(640×360) 480P(854×480)