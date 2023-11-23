import matplotlib
import matplotlib.pyplot as plt

import argparse
import os
import datetime
import pathlib
import random
import json
import numpy as np
import math
import pandas as pd

import torch

import sys
sys.path.append('code/')
from linear_utils import linear_model
from train_utils import save_config
import two_layer
import wandb


def get_args():
    parser = argparse.ArgumentParser(description='CLI parameters for training')
    parser.add_argument('--config', type=str, default='', metavar='CONFIG',
                        help='Config file')
    parser.add_argument('--root', type=str, default='', metavar='DIR',
                        help='Root directory')
    parser.add_argument('-t', '--iterations', type=int, default=1e4, metavar='ITERATIONS',
                        help='Iterations (default: 1e4)')
    parser.add_argument('-n', '--samples', type=int, default=100, metavar='N',
                        help='Number of samples (default: 100)')
    parser.add_argument('--print-freq', type=int, default=100,
                        help='CLI output printing frequency (default: 1000)')
    parser.add_argument('--gpu', type=int, default=None,
                        help='Number of GPUS to use')
    parser.add_argument('--disable-cuda', action='store_true', default=False,
                        help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('-d', '--dim', type=int, default=50, metavar='DIMENSION',
                        help='Feature dimension (default: 50)')
    parser.add_argument('--hidden', type=int, default=50, metavar='DIMENSION',
                        help='Hidden layer dimension (default: 200)')
    parser.add_argument('--batch-norm', action='store_true', default=False,
                        help='Use batch norm')
    parser.add_argument('--no-bias', action='store_true', default=False,
                        help='Do not use bias')
    parser.add_argument('--linear', action='store_true', default=False,
                        help='Linear activation function')
    parser.add_argument('--sigmas', type=str, default=None,
                        help='Sigmas')
    parser.add_argument('--sigma_noise', nargs='*', type=float, default=0.0,
                        help='Output noise.')
    parser.add_argument('--beta', nargs='*', type=float, default=None,
                        help='True model parameters.')
    parser.add_argument('--coupled_noise', action='store_true', default=False,
                        help='Couple noise in output to large eigenvalues.')
    parser.add_argument('-r', '--s-range', nargs='*', type=float,
                        help='Range for sigmas')
    parser.add_argument('-w', '--scales', nargs='*', type=float,
                        help='scale of the weights')
    parser.add_argument('--first_layer_lr', type=float, default=1e-4, metavar='FIRST LR',
                        help='First layer lr')
    parser.add_argument('--lr_factor', type=float, default=1e-4, metavar='LR RATIO',
                        help='Factor with which first layer lr i multiplied to obtain second layer lr')
    parser.add_argument('--fixed_last_layer_lr', action='store_true', default=False,
                        help='If true, learning rate of last layer is not relative but fixed to lr_factor.')
    parser.add_argument('--normalized', action='store_true', default=False,
                        help='normalize sample norm across features')
    parser.add_argument('--risk-loss', type=str, default='MSE', metavar='LOSS',
                        help='Loss for validation')
    parser.add_argument('--jacobian', action='store_true', default=False,
                        help='compute the SVD of the jacobian of the network')
    parser.add_argument('--save-results', action='store_true', default=False,
                        help='Save the results for plots')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Plot the results')
    parser.add_argument('--eigen', action='store_true', default=False,
                        help='Compute eigenvalue')
    parser.add_argument('--pcs', type=int, default=None, 
                        help='Number of PCs to use in data.')
    parser.add_argument('--transform-data', action='store_true', default=False, 
                        help='Use data in transformed space')
    parser.add_argument('--kappa', type=float, default=None, 
                        help='Scaling factor for eigenvalues of input.')
    parser.add_argument('--low-rank-eval', action='store_true', default=False, 
                        help='Evaluate performance of low-rank train data.')
    parser.add_argument('--ind-eval', action='store_true', default=False, 
                        help='Evaluate performance of individual training examples.')
    parser.add_argument('--weight-eval', action='store_true', default=False, 
                        help='Evaluate MSE of weights (linear model).')
    parser.add_argument('--save-weights', action='store_true', default=False, 
                        help='Save model weights.')
    parser.add_argument('--details', type=str, metavar='N',
                        default='no_detail_given',
                        help='details about the experimental setup')
    parser.add_argument('--num-layers', type=int, default=2, 
                        help='number of model layers (1, 2 or 5)')
    parser.add_argument('--freeze-layer', type=int, default=None, 
                        help='Freezing model layer.')
    parser.add_argument('--fixed-weight-init', action='store_true', default=False, 
                        help='If initial weights should be set to fixed values, specified by args.scales.')
    parser.add_argument('--scaling-layer', action='store_true', default=False,
                        help='Use ScalingLayer as last layer (for analysis).')
 

    args = parser.parse_args(sys.argv[1:])

    # directories
    root = pathlib.Path(args.root) if args.root else pathlib.Path.cwd().parent

    current_date = str(datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
    args.outpath = (pathlib.Path.cwd().parent / 'results' / 'two_layer_nn' / current_date)

    if args.save_results:
        args.outpath.mkdir(exist_ok=True, parents=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    args.device = 'cuda' if (not args.disable_cuda and torch.cuda.is_available()) else 'cpu'
    print(args.device)

    if args.fixed_last_layer_lr:
        args.lr = [args.first_layer_lr, args.lr_factor]
    else:
        args.lr = [args.first_layer_lr, args.first_layer_lr*args.lr_factor]
    
    if len(args.sigma_noise) == 1:
        args.sigma_noise = args.sigma_noise[0]

    return args

if __name__ == "__main__":
    args = get_args()
    two_layer.main(args)