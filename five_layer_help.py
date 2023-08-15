#!/usr/bin/env python
# coding: utf-8

# %%
import os
import random
import shutil
import pathlib
import json
import time
import datetime
import warnings
import numpy as np
from IPython.core.debugger import set_trace
from PIL import Image
import pickle
import argparse
import torch

import five_layer

import sys
repo_root = os.path.join(os.getcwd(), './code/')
sys.path.append(repo_root)


# %%
def get_args():

    # get CLI parameters
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', type=str, default='', metavar='FILE',
                        help='JSON file containing the configuration dictionary')

    parser = argparse.ArgumentParser(description='CLI parameters for training')
    parser.add_argument('--root', type=str, default='', metavar='DIR',
                        help='Root directory')
    parser.add_argument('--main', type=str, default='cifar.npz', metavar='FILE',
                        help='Main file')
    parser.add_argument('--test', type=str, default='cifar.npz', metavar='FILE',
                        help='Test file')
    parser.add_argument('--sub', type=str, default='',
                        help='Sub file')
    parser.add_argument('--epochs', type=int, default=90, metavar='EPOCH',
                        help='Epochs (default: 90)')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch (default: 0)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='BATCH',
                        help='Batch size (default: 128)')
    parser.add_argument('--print-freq', type=int, default=1000,
                        help='CLI output printing frequency (default: 1000)')
    parser.add_argument('--workers', type=int, default=4, metavar='WORKERS',
                        help='Number of workers (default: 4)')
    parser.add_argument('--secure-checkpoint', action='store_true', default=False,
                        help='Checkpoint in outdir rather than rootdir.')  
    parser.add_argument('--gpu', type=int, default=None,
                        help='Number of GPUS to use')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')                        
    parser.add_argument('--model', type=str, default='resnet',
                        help='Model architecture')   
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Use pretrained model')
    parser.add_argument('--model-config')
    parser.add_argument('--num-planes', type=int, default=64, metavar='WIDTH',
                        help='Model width (default: 64)')
    parser.add_argument('--layer-names', action='store_true', default=False,
                        help='Initialize model with names for the layers.') 
    parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "sgd")')
    parser.add_argument('--loss', type=str, default='cross_entropy', metavar='LOSS',
                        help='loss function (default: cross entropy')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')                        
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)') # TODO: not used?                      
    parser.add_argument('--lrdecay', '--lrdr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)') # TODO: not used?                      
    parser.add_argument('--decay-epochs', type=int, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--decay-max-epochs', type=int, default=70, metavar='N',
                        help='max number of epochs to decay LR')                        
    parser.add_argument('--use-inverse-sqrt-lr', action='store_true', default=False,
                        help='Use inverse square-root learning rate decay')   
    parser.add_argument('--inverse-rate', type=float, default=512.0, metavar='RATE',
                        help='Inverse square-root decay rate (default: 512)')
    parser.add_argument('--schedule-lr', action='store_true', default=False,
                        help='Use lr scheduler')  
    parser.add_argument('--norandomcrop', action='store_true', default=False,
                        help='Disable random crop augmentation')  
    parser.add_argument('--norandomflip', action='store_true', default=False,
                        help='Disable random flip augmentation')  
    parser.add_argument('--train-size', type=int, default=0,
                        help='Size of the random training subset to use (default: all)')
    parser.add_argument('--compute-variance', action='store_true', default=False,
                        help='Compute the bias and variance')
    parser.add_argument('--trials', type=int, default=0,
                        help='How many training splits to use (default: None)')
    parser.add_argument('--split', type=int, default=0,
                        help='Current training split to use (default: first)')
    parser.add_argument('--trial-file', default='', type=str, metavar='FILE',
                        help='Resume from existing trial file')
    parser.add_argument('--select-classes', default=[], type=int, nargs='*',
                        help='Selected subset of classes (default: all)')              
    parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                        help='Number of label classes (default: 10)')
    parser.add_argument('--inject-noise', type=float, default=0.0, metavar='NOISE',
                        help='symmetric noise level for injection')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Evaluate performance and quit')   
    parser.add_argument('--resume', default='', type=str, metavar='RESUME',
                        help='Resume from checkpoint')
    parser.add_argument('--track-correct', action='store_true', default=False,
                        help='Track indices of correctly classified examples')   
    parser.add_argument('--scale-lr', type=float, default=None, metavar='SCALE',
                        help='Scale the learning rate of the fully connected layer')
    parser.add_argument('--initial-lr')
    parser.add_argument('--scale-weights')
    parser.add_argument('--initial-lr-decay', default=[],
                        help='Initial learning rate decay')
    parser.add_argument('--compute-jacobian', action='store_true', default=False,
                        help='Record the Jacobian norm')   
    parser.add_argument('--compute-jacobian-svd', action='store_true', default=False,
                        help='Compute the SVD of the Jacobian')  
    parser.add_argument('--average-batches', action='store_true', default=False,
                        help='Whether to average the SVD of the Jacobian across batches')  
    parser.add_argument('--track-weights', action='store_true', default=False,
                        help='Record the norm of the weights')   
    parser.add_argument('--details', type=str, metavar='N', nargs='*',
                        default=['no', 'details', 'given'],
                        help='details about the experimental setup')

    def _parse_args():
        # Do we have a config file to parse?
        args_config, remaining = config_parser.parse_known_args()
        if args_config.config:
            with open(args_config.config) as f:
                cfg = json.load(f)
                parser.set_defaults(**cfg)

        # The main arg parser parses the rest of the args, the usual
        # defaults will have been overridden if config file specified.
        args = parser.parse_args(remaining)
        return args

    # set parameters
    args = _parse_args()

    # directories
    args.root = pathlib.Path(args.root) if args.root else pathlib.Path.cwd()

    # details
    args.details = ' '.join(args.details)

    # size and number of classes
    if args.select_classes:
        args.num_classes = len(args.select_classes)
    if not isinstance(args.model_config, dict):
        args.model_config = {'num_planes': args.num_planes}
        args.model_config.update({'num_classes': args.num_classes})
        if args.layer_names:
            args.model_config.update({'layer_names': args.layer_names})

    args.scale_lr = {'17': args.lr * args.scale_lr} if args.scale_lr else {}
    
    return args


if __name__ == "__main__":

    args = get_args()

    # rescale weights
    scale_weights = False

    # %%

    #current_date = str(datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
    #args.outpath = (pathlib.Path.cwd() / 'results' / args.model / 
    #              os.path.splitext(args.main)[0]) # / current_date)

    #if not args.outpath.exists():
    #    args.outpath.mkdir(parents=True)
    #else:
    #    i = 1
    #    new_outpath = args.outpath.parent / (args.outpath.name + '_' + str(i))
    #    while new_outpath.exists():
    #        i += 1
    #        new_outpath = args.outpath.parent / (args.outpath.name + '_' + str(i))
    #        assert count < 100, "Could not find an appropriate output path!"
    #    args.outpath = new_outpath
    #    args.outpath.mkdir(parents=True)
        

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                    'This will turn on the CUDNN deterministic setting, '
                    'which can slow down your training considerably! '
                    'You may see unexpected behavior when restarting '
                    'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                    'disable data parallelism.')

    #ngpus_per_node = torch.cuda.device_count()
    # Simply call main_worker function
    five_layer.main_worker(args.gpu, args)#ngpus_per_node, args)
