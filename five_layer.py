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

import sys
repo_root = os.path.join(os.getcwd(), './code/')
sys.path.append(repo_root)
from train_utils import (CandidateDataset, AverageMeter, 
                        save_checkpoint, save_config, 
                        adjust_learning_rate, cross_entropy_split, clear_gradients)
from proj_utils import get_jacobian_prod, get_jacobian_svd, get_weights

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import model_select

import wandb


# %%
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

best_acc1 = 0

# holders for epoch-wise bias-variance calculations
OUTPUTS_SUM_LIST = []
OUTPUTS_SUMNORMSQUARED_LIST = []


# NOTE: I (Amanda) have restructured the code a bit. Hope that I haven't fucked up any important ordering.
def main_worker(gpu, args):
    global best_acc1
    
    args.outpath = get_result_dir(args.model)
    wandb.init(project="double_descent_five_layer", name=get_run_name(args), config=args)
    

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        torch.cuda.set_device(args.gpu)

    train_loader, val_loader, val_loader2 = get_data(args)
    model = get_model(args)

    if args.loss in ['l2', 'l2_squared', 'squared', 'MSE']:
        args.loss = 'l2'
    
    criterion, optimizer = get_training_setup(model, args)


    cudnn.benchmark = True
    
    # TODO: I would like to move the content of following function to get_model and get_training_setup, but unsure if the particular order is important
    model, optimizer = scale_weights_and_lr(model, optimizer, args)
    
    if args.schedule_lr:
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 
                                                      args.lr / 100, args.lr)
    else: 
        scheduler = None
    
    
    train_model(train_loader, val_loader, val_loader2, model, criterion, optimizer, args)
    
def scale_weights_and_lr(model, optimizer, args):
    
    # TODO: scaling the weights of the model manually
    if args.scale_weights:
        scale_dict = {}
        for cur_l, cur_w in enumerate(cur_weights):
            if not (cur_w.ndim > 2):
                continue
            scale_dict['layer_' + str(layer_idx[cur_l])] = np.linalg.norm(cur_w.flatten()).item()
        rescale_weights(model, scale_dict)
        
        
    if args.scale_lr:
        if not args.opt.lower() == 'sgd':
            raise ValueError('SGD must be selected when learning rates are scaled!')
        if isinstance(args.scale_lr, dict):
            opt_lr_dict = {k: v for k, v in args.scale_lr.items()}
        else:
            scale_dict = {}
            for cur_l, cur_w in enumerate(cur_weights):
                if not (cur_w.ndim > 2):
                    continue
                scale_dict['layer_' + str(layer_idx[cur_l])] = np.linalg.norm(cur_w.flatten()).item()

            opt_lr_dict = get_lr_scales(model, args.lr, scale_dict)
        
        param_setup = [{'params': cur_lay, 'lr': opt_lr_dict[str(i)]} 
                       if (str(i) in opt_lr_dict)
                       else {'params': cur_lay}
                       for i, cur_lay in enumerate(model))] # model.parameters()
                       if 'weight' in dir(cur_lay)]
        args.initial_lr = [{'lr': opt_lr_dict[str(i)]} 
                           if (str(i) in opt_lr_dict)
                           else {'lr': args.lr}
                           for i, cur_lay in enumerate(model)] # model.parameters()
                           if 'weight' in dir(cur_lay)]
                          
        for p in param_setup:
            print(p)
        optimizer = torch.optim.SGD(param_setup, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        

    # TODO: testing scaled initialization
    if 'testing_scaled_initialization' in args.details:
        print('Warning: Scaling the weights of the linear layer by {}!'.format(float(args.details.split()[-1])))
        if args.resume:
            torch.nn.init.kaiming_uniform_(model[-1].weight, a=np.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(model[-1].weight)
            bound = 1 / np.sqrt(fan_in)
            with torch.no_grad():
                model[-1].bias.uniform_(-bound, bound)
        scale_initialization(model, [0,3,7,11], float(args.details.split()[-1]))

    return model, optimizer

def get_data(args):
    # Data loading code
    main_file = args.root / args.main
    test_file = args.root / args.test
    if args.sub:
        sub = args.root / args.sub
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    train_trans_list = []
    if not args.norandomcrop:
        train_trans_list.append(transforms.RandomCrop(32, padding=4))
    if not args.norandomflip:
        train_trans_list.append(transforms.RandomHorizontalFlip())
    train_trans_list = train_trans_list + [transforms.ToTensor(), normalize]
    
    
    train_dataset = CandidateDataset(
        main_file,
        transforms.Compose(train_trans_list),
        save_true = args.evaluate_gradients
    )
    
    test_dataset = CandidateDataset(test_file, 
                                    transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize,
                                    ]), 
                                    train=False)
    
    if args.sub:
        sub_dataset = CandidateDataset(sub_file, 
                                        transforms.Compose([
                                            transforms.ToTensor(),
                                            normalize,
                                        ]), 
                                        train=False)
    

    args.kmax = min(5, len(args.select_classes) - 1) if args.select_classes else 5
    if args.train_size or args.select_classes:
        if not args.select_classes:
            args.select_classes = list(range(args.num_classes))
        sel_idx = []
        for lbl in args.select_classes:
            lbl_idx = [i for i, t in enumerate(train_dataset.targets) if t == lbl]
            sel_idx += random.sample(lbl_idx, (args.train_size if args.train_size else len(lbl_idx)))
        train_dataset.samples = train_dataset.samples[sel_idx]
        train_dataset.targets = train_dataset.targets[sel_idx]
        for cur_idx, cur_cls in enumerate(args.select_classes):
            train_dataset.targets[train_dataset.targets==cur_cls] = cur_idx
        
        sel_idx = []
        for lbl in args.select_classes:
            lbl_idx = [i for i, t in enumerate(test_dataset.targets) if t == lbl]
            sel_idx += lbl_idx
        test_dataset.samples = test_dataset.samples[sel_idx]
        test_dataset.targets = test_dataset.targets[sel_idx]
        for cur_idx, cur_cls in enumerate(args.select_classes):
            test_dataset.targets[test_dataset.targets==cur_cls] = cur_idx
    

    # split the training set for trials
    if args.trials:
        if not args.select_classes:
            args.select_classes = list(range(args.num_classes))
        sel_idx = []
        for lbl in args.select_classes:
            lbl_idx = [i for i, t in enumerate(train_dataset.targets) if t == lbl]
            sel_idx += list(np.split(np.random.permutation(lbl_idx), args.trials)[args.split])

        train_dataset.samples = train_dataset.samples[sel_idx]
        train_dataset.targets = train_dataset.targets[sel_idx]

    
    # Inject symmetric noise to training set
    if args.inject_noise:
        im_per_class = int(len(train_dataset) / args.num_classes)
        noisy_labels = np.zeros((len(train_dataset),), dtype=int)
        num_shuffle = int(im_per_class * (args.inject_noise / (args.num_classes - 1)))
        for i in range(args.num_classes):
            noisy_idx = []
            cur_idx = [idx for idx, label in enumerate(train_dataset.targets) if label==i]
            shuffled_idx = random.sample(cur_idx, len(cur_idx))
            for r in range(args.num_classes):
                noisy_idx += [r for idx in shuffled_idx[im_per_class - (r+1)*num_shuffle:im_per_class - r*num_shuffle]]
            noisy_idx += [i for idx in shuffled_idx[:im_per_class - args.num_classes*num_shuffle]]
            noisy_labels[cur_idx] = np.array(noisy_idx)
        train_dataset.targets = noisy_labels   
        
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    if args.sub:
        val_loader2 = torch.utils.data.DataLoader(
            sub_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:
        val_loader2 = None
            
    return train_loader, val_loader, val_loader2


def get_model(args):
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.model))
        model = models.__dict__[args.model](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.model))
        model = model_select.BaseModel.create(args.model, **args.model_config)
        
        
    if args.gpu is not None:
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.model.startswith('alexnet') or args.model.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
            
    return model 


def get_training_setup(model, args):
    # define loss function (criterion) and optimizer
    
    if args.loss in ['cross', 'cross_entropy', 'entropy']:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    
    elif args.loss in ['l2', 'l2_squared', 'squared', 'MSE']:
        print('[INFO] Using MSE loss function instead of Cross Entropy.')
        args.loss = 'l2'
        criterion = nn.MSELoss().cuda(args.gpu)

    if args.opt.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.opt.lower() == 'adam':
        print('[INFO] Using Adam optimizer instead of SGD.')
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                    weight_decay=args.weight_decay)
    elif args.opt.lower() == 'lbfgs':
        print('[INFO] Using LBFGS optimizer instead of SGD.')
        optimizer = torch.optim.LBFGS(model.parameters(), args.lr,
                                      history_size=20
                                     )
    else:
        raise ValueError('Incorrect optimizer selection {}'.format(args.opt))

    if args.initial_lr:
        param_setup = [{'params': cur_lay.parameters()} 
                       for i, cur_lay in enumerate(model)
                       if 'weight' in dir(cur_lay)]
        optimizer = torch.optim.SGD(param_setup, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        
    return criterion, optimizer
   

def train_model(train_loader, val_loader, val_loader2, model, criterion, optimizer, args):

    best_acc1 = 0.0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            

    if args.evaluate:
        save_config(args)
        validate(val_loader, model, criterion, args)
        return
    
    if args.compute_jacobian:
        assert not args.compute_jacobian_svd, "Jacobian prod and Jacobian SVD cannot be set at the same time."
        gvec = (torch.randn((1, args.num_classes)) / len(train_dataset)).cuda(args.gpu, non_blocking=True)
    
    if args.compute_jacobian_svd:
        sv, vconv, vfc = get_jacobian_svd(train_loader, model, args, average_batches=args.average_batches)

        svd_file = os.path.join(args.outpath, get_run_name(args) + '_jacobian_svd.npz')
        np.savez(svd_file, sv=sv, vconv=vconv, vfc=vfc)
        save_config(args)
        return


    # TODO: tracking weights of the model (NOTE: this does not work if model is DataParallel (gpu unspecified)
    if args.track_weights:
        layer_idx = [i for i, cl in enumerate(model) if 'weight' in dir(cl)] # model.parameters()
        cur_weights = get_weights(model, layer_idx)
        #if args.track_weights == 'filters':
        #    filter_w_file = os.path.join(args.outpath, get_run_name(args) + '_filter_weights.pickle')
        #    filter_w_dict = {('layer_'+str(l)): [] for i, l in enumerate(layer_idx) 
        #                     if cur_weights[i].ndim > 2}
        #if args.track_weights == 'norm':
        w_norm_dict = {('layer_'+str(l)): 0 for i, l in enumerate(layer_idx) 
                         if cur_weights[i].ndim > 1}


    save_config(args, get_run_name(args))
    train_log = []
    log_file = os.path.join(args.outpath, get_run_name(args) + '_log.json')

    for epoch in range(args.start_epoch, args.epochs):
        if (epoch < args.decay_max_epochs) and (not args.schedule_lr):
            adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)
        
        epoch_log = {'epoch': epoch}
        
        # update learning rate with scheduler
        if args.schedule_lr:
            scheduler.step()

        # evaluate on validation set
        tr_acc1, tr_acc5, _ = validate(train_loader, model, criterion, args)
        epoch_log.update({'train': {'acc1': tr_acc1.cpu().numpy().item(), 
                                    'acc5': tr_acc5.cpu().numpy().item()}})
        
        acc1, acc5, test_loss = validate(val_loader, model, criterion, args)
        epoch_log.update({'test': {'acc1': acc1.cpu().numpy().item(), 
                                   'acc5': acc5.cpu().numpy().item(),
                                   'loss': test_loss}})
                                   
        if args.evaluate_gradients:
            grad_dep, grad_ind = compute_gradients(train_loader, model, args) 
            epoch_log.update({'grads': {'grad_dep': grad_dep, 
                                        'grad_ind': grad_ind}})
        
        if args.sub:
            dum_acc1, dum_acc5, _ = validate(val_loader2, model, criterion, args)
            epoch_log.update({'subset': {'acc1': dum_acc1.cpu().numpy().item(), 
                                         'acc5': dum_acc5.cpu().numpy().item()}})

        
        # compute the bias and variance
        if args.compute_variance:

            if args.trial_file:
                cur_file = os.path.join(args.trial_file, '{}.pickle'.format(epoch))
                with open(cur_file, 'rb') as fn:
                    outputs_sum, outputs_sumnormsquared, test_loss_sum = pickle.load(fn)
            else:
                outputs_sum = torch.Tensor(len(test_dataset), args.num_classes).zero_().cuda(args.gpu)
                outputs_sumnormsquared = torch.Tensor(len(test_dataset)).zero_().cuda(args.gpu)
                test_loss_sum = 0
            
            test_loss_sum += test_loss
            bias2, variance, outputs_sum, outputs_sumnormsquared = compute_bias_variance(model, val_loader, args, outputs_sum, outputs_sumnormsquared)
            variance_unbias = variance * args.trials / (args.trials - 1.0)
            bias2_unbias = test_loss_sum / (args.split + 1) - variance_unbias
        
            epoch_log.update({'bias': bias2_unbias.item(), 'variance': variance_unbias.item()})
            with open(args.outpath / get_run_name(args) + '_{}.pickle'.format(epoch), 'wb') as fn:
                pickle.dump([outputs_sum, outputs_sumnormsquared, test_loss_sum], fn)

        # compute the jacobian of the network
        if args.compute_jacobian:
            jTg = get_jacobian_prod(train_loader, model, criterion, gvec, args)
            epoch_log.update({'J_norm': {str(k): v.item() for k, v in enumerate(jTg)}})
        
        # TODO: tracking the weights of the layers
        if args.track_weights:
            w_change_dict = {('layer_'+str(l)): 0 for l in layer_idx}
            new_weights = get_weights(model, layer_idx)
            
            #if args.track_weights == 'norm':
            for cur_l, cur_w in enumerate(new_weights):
                if not (cur_w.ndim > 1):
                    continue
                w_norm_dict['layer_' + str(layer_idx[cur_l])] = np.linalg.norm(cur_w.flatten()).item()
            epoch_log.update({'w_norm': {k: v for k, v in w_norm_dict.items()}})
                
            #else:
            #    for cur_l in range(len(layer_idx)):
            #        cur_change = new_weights[cur_l] - cur_weights[cur_l]

            #        if args.track_weights == 'filters':
            #            if cur_change.ndim > 2:
            #                cur_change = np.mean(cur_change, axis=(2,3))
            #                filter_w_dict['layer_' + str(layer_idx[cur_l])].append(np.absolute(cur_change))

            #        chng = np.absolute(np.mean(cur_change))
            #        w_change_dict['layer_' + str(layer_idx[cur_l])] = chng.item()

            #    epoch_log.update({'weight_change': {k: v for k, v in w_change_dict.items()}})

            #    if args.track_weights == 'filters':
            #        with open(filter_w_file, 'wb') as fn:
            #            pickle.dump({k: np.stack(v) for k, v in filter_w_dict.items()}, fn)

            #   cur_weights = [wh for wh in new_weights]
            #    new_weight = None
        
        train_log.append(epoch_log)
        with open(log_file, 'w') as fn:
            json.dump(train_log, fn, indent=2)
        
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1 and (not torch.isnan(acc1)) and (not torch.isinf(acc1))
        best_acc1 = max(acc1, best_acc1)
            

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.model,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename=(args.outpath if args.secure_checkpoint else None))
        
        #if np.isnan(acc1):
        #    break


def compute_bias_variance(net, testloader, args, outputs_sum, outputs_sumnormsquared):
    net.eval()
    bias2 = 0
    variance = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)
            targets_onehot = torch.FloatTensor(targets.size(0), args.num_classes).cuda(args.gpu)
            targets_onehot.zero_()
            targets_onehot.scatter_(1, targets.view(-1, 1).long(), 1)
            outputs = net(inputs)
            outputs = F.softmax(outputs, dim=1)
            outputs_sum[total:(total + targets.size(0)), :] += outputs
            outputs_sumnormsquared[total:total + targets.size(0)] += outputs.norm(dim=1) ** 2.0

            bias2 += (outputs_sum[total:total + targets.size(0), :] / (args.split + 1) - targets_onehot).norm() ** 2.0
            variance += outputs_sumnormsquared[total:total + targets.size(0)].sum()/(args.split + 1) - (outputs_sum[total:total + targets.size(0), :]/(args.split + 1)).norm() ** 2.0
            total += targets.size(0)

    return bias2 / total, variance / total, outputs_sum, outputs_sumnormsquared


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        if args.use_inverse_sqrt_lr:
            assert not (args.decay_max_epochs > 0), "lr scheduler crash, step and inverse sqrt lr can't be mutually set"
            
            for cur_p, param_group in enumerate(optimizer.param_groups):
                d_rate = (args.initial_lr_decay[cur_p]['decay']
                          if args.initial_lr_decay
                          else args.inverse_rate)
                base_lr = (args.initial_lr[cur_p]['lr']
                          if args.initial_lr
                          else args.lr)
                lr = base_lr / np.sqrt(1 + (epoch*len(train_loader) + i)/d_rate)
                param_group['lr'] = lr

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        
        if args.loss == 'l2':
            zero_mat = np.zeros((len(target), args.num_classes), dtype=int)
            zero_mat[list(range(len(target))), target] = 1
            targetl2 = torch.from_numpy(zero_mat).float()
            targetl2 = targetl2.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        
        # for LBFGS
        if args.opt.lower() == 'lbfgs':
            def closure():
                optimizer.zero_grad()
                output = model(input)
                loss = criterion(output, target)
                loss.backward()
                return loss
            
            optimizer.step(closure)
            
        else:
            # compute output
            output = model(input)
            if args.loss == 'l2':
                loss = criterion(output, targetl2)
            else:
                loss = criterion(output, target) 

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, args.kmax))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    if args.track_correct:
        corr_dict = {'correct':[]}
    
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            
            if args.loss == 'l2':
                zero_mat = np.zeros((len(target), args.num_classes), dtype=int)
                zero_mat[list(range(len(target))), target] = 1
                targetl2 = torch.from_numpy(zero_mat).float()
                targetl2 = targetl2.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            if args.loss == 'l2':
                loss = criterion(output, targetl2)
            else:
                loss = criterion(output, target)
            
            # record correctly classified examples
            if args.track_correct:
                correct = accuracy(output, target, topk=(1, args.kmax), track=True)
                corr_dict['correct'] += [(i*args.batch_size) + idx for idx, is_corr in 
                                         enumerate(correct) if is_corr]

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, args.kmax))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        # Record the indices of the correctly classified images
        if args.track_correct:
            fname, ext = str(args.outpath).split('.')
            corrfile = fname + '_corr.json'
            with open(corrfile, 'w') as f:
                json.dump(corr_dict, f, indent=2)
            return

    return top1.avg, top5.avg, losses.avg

def evaluate_weight_norms(model):

    weight_dict = {}
    for i, param in enumerate(model.parameters()):
        print(param.shape)
        weight_dict["layer_" + str(i)] = torch.norm(param)

def compute_gradients(val_loader, model, args):
    # TODO: clean this up
    
    model.eval()
    val_loader.dataset.set_return_true(True) # This will return true targets when fetching items
    
    grad_dep, grad_ind = {}, {}
    for i, (inputs, targets, true_targets) in enumerate(val_loader):
    
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu, non_blocking=True)
       
        targets = targets.cuda(args.gpu, non_blocking=True)
        true_targets = true_targets.cuda(args.gpu, non_blocking=True)

        # compute output
        outputs = model(inputs)
        one_hot_targets = nn.functional.one_hot(targets, args.num_classes)
        
        # Split into unpermuted and permuted examples
        outputs_unperm, targets_unperm = outputs[targets == true_targets, :], one_hot_targets[targets == true_targets, :]
        outputs_perm, targets_perm = outputs[targets != true_targets, :], one_hot_targets[targets != true_targets, :]

        for j, (key, (outs, targs)) in enumerate(zip(['unperm', 'perm'], [(outputs_unperm, targets_unperm), (outputs_perm, targets_perm)])):
            if targs.shape[0] > 0:
            
                if outs.ndim == 1:
                    outs = outs.reshape(1, -1)
                if targs.ndim == 1:
                    targs = targs.reshape(1, -1)
            
                # Calculate loss
                loss_dep, loss_ind = cross_entropy_split(outs, targs)
        
                # Calculate gradients for label dependent part
                loss_dep.backward(retain_graph=True)
                
                # Accumulate gradients
                grad_dict = {}
                for k, param in enumerate(model.parameters()):
                    grad_dict['layer_' + str(k)] = param.grad.detach()
                
                if key not in grad_dep:
                    grad_dep[key] = grad_dict.copy()
                else:
                    for key2 in grad_dep[key]:
                        grad_dep[key][key2] += grad_dict[key2]
                
                # Clear gradients
                clear_gradients(model)
                
                # Just a check
                for param in model.parameters():
                    assert param.grad is None
                      
                # Calculate gradients for label independent part
                loss_ind.backward(retain_graph=True)
                
                # Accumulate gradients
                grad_dict = {}
                for k, param in enumerate(model.parameters()):
                    grad_dict['layer_' + str(k)] = param.grad.detach()
                
                if key not in grad_ind:
                    grad_ind[key] = grad_dict.copy()
                else:
                    for key2 in grad_ind[key]:
                        grad_ind[key][key2] += grad_dict[key2]
                 
                # Clear gradients
                clear_gradients(model)
                
                # Just a check
                for param in model.parameters():
                    assert param.grad is None
     
            del loss_dep, loss_ind
            
        del outputs
    
    model.train()
    val_loader.dataset.set_return_true(False)
    
    # Compute norms
    for key in grad_dep:
        
        if key == 'unperm':
            num_samples = (val_loader.dataset.targets == val_loader.dataset.true_targets).sum()
        elif key == 'perm':
            num_samples = (val_loader.dataset.targets != val_loader.dataset.true_targets).sum()

        for key2 in grad_dep[key]:
            grad_dep[key][key2] = torch.norm(grad_dep[key][key2] / num_samples).cpu().numpy().item()
            grad_ind[key][key2] = torch.norm(grad_ind[key][key2] / num_samples).cpu().numpy().item()
                
    return grad_dep, grad_ind

def accuracy(output, target, topk=(1,), track=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        # return indices of the correctly classified examples instead of accuracy.
        if track:
            return correct[:1].view(-1).cpu().numpy()

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
        

def get_run_name(args):
    # TODO: make this nicer
    lr2 = args.scale_lr['17'] if args.scale_lr else args.lr
    run_name = f'lr={args.lr}_{lr2}'
    
    if not args.norandomflip:
        run_name += '_r_flip'
    if not args.norandomcrop:
        run_name += '_r_crop'
        
    if args.weight_decay > 0.0:
        run_name += f'_wd={args.weight_decay}'
        
    if args.lrdecay < 1.0:
        run_name += f'_ld={args.lrdecay}'
        
    if args.track_weights:
        run_name += '_tw'
        
    if args.momentum == 0.0:
        run_name += "_no_momentum"
        
    if args.use_inverse_sqrt_lr:
        run_name += "_inv_sq_lr"
        
    return run_name
    

def get_result_dir(model="mcnn"):

    if model == "2nn":
        base_dir = pathlib.Path.cwd() / "results/two_layer_classification_results"
    else:
        base_dir = pathlib.Path.cwd() / "results/five_layer_results"
        
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)  # (io.get_checkpoint_root())
    
    return base_dir

def get_result_path(args):
    run_name = get_run_name(args)
    base_dir = get_result_dir(args.model)
    
    result_path = os.path.join(base_dir, run_name + ".csv")
    return result_path


def save_results(args, risks, losses=None):
    result_path = get_result_path(args)
    data = pd.DataFrame(risks)
    if losses is not None:
        data[1] = losses
    data.to_csv(result_path, header=False, index=False)

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
                        help='LR decay rate (default: 0.1)') #TODO: not used?
    parser.add_argument('--lrdecay', '--lrdr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')     
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
    parser.add_argument('--evaluate-gradients', action='store_true', default=False,
                        help='whether or not to evaluate and save gradients (for ce loss)')
    use_inverse_sqrt_lr
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
    
        if model == "2nn":
            args.model_config = {'input_size': 32*3*3}
            args.model_config.update({'hidden_size': 1000})
            args.model_config.update({'num_classes': args.num_classes})
        
        else:
            args.model_config = {'num_planes': args.num_planes}
            args.model_config.update({'num_classes': args.num_classes})
        if args.layer_names:
            args.model_config.update({'layer_names': args.layer_names})

        
    args.scale_lr = {'17': args.lr * args.scale_lr} if args.scale_lr else {}


# %%
if __name__ == "__main__":

    args = get_args()

    # rescale weights
    scale_weights = False

    # %%

    current_date = str(datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
    args.outpath = (pathlib.Path.cwd() / 'results' / args.model / 
                    os.path.splitext(args.main)[0] / current_date)

    if not args.outpath.exists():
        args.outpath.mkdir(parents=True)
    else:
        i = 1
        new_outpath = args.outpath.parent / (args.outpath.name + '_' + str(i))
        while new_outpath.exists():
            i += 1
            new_outpath = args.outpath.parent / (args.outpath.name + '_' + str(i))
            assert count < 100, "Could not find an appropriate output path!"
        args.outpath = new_outpath
        args.outpath.mkdir(parents=True)
        

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
    main_worker(args.gpu, args)
