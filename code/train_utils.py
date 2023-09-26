import pathlib
import shutil
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision


class CandidateDataset(Dataset):
    """Candidate dataset."""
    
    def __init__(self, pathname, transform=None, train=True, save_true=False):
        """
        Args:
            pathname (pathlib.Path): Path to the npz file.
            transform (callable, optional): Optional transform to be applied on a sample.
            train (boolean): Extract train (True) or test (False) set from the file.
            save_true (boolean): Saving a copy of targets (in case of injected noise).
        """
        
        self.save_true = save_true
        self.return_true = False

        self.samples, self.targets = np_loader(pathname.resolve(), train=train)
        
        if self.save_true:
            self.true_targets = self.targets.copy()
        
        self.transform = transform
        
    def set_return_true(self, return_true):
    
        assert self.save_true, "Can not return true targets, as they are not saved"
        
        self.return_true = return_true
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        
 
        sample, target = self.samples[index], self.targets[index]
            
        sample = Image.fromarray(np.moveaxis(sample, 0, -1))
        
        if self.transform is not None:
            sample = self.transform(sample)
            
        # TODO: Target transform.
        
        # TODO: THIS WILL DESTROY GRADIENT EVALUATION
        #if self.return_true:
        #    return sample, target, self.true_targets[index]
        #else:
        return sample, target
    
def np_loader(filename, train=True):
        #data = np.load(filename)
    if train:
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True)

        #samples = data['X_train'].transpose(0, 3, 1, 2)
        #targets = data['y_train']
    else:
        #samples = data['X_test'].transpose(0, 3, 1, 2)
        #targets = data['y_test']
        dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True)
    samples, targets = dataset.data.transpose(0, 3, 1, 2), dataset.targets        
    return samples, targets


def prune_data(Xs, k):
    U, S, Vh = np.linalg.svd(Xs, full_matrices=True)
    
    if k >= S.shape[0]:
        return Xs
    
    Xs_pruned = torch.tensor(np.dot(U[:, :k] * S[:k], Vh[:k, :]))
    
    return Xs_pruned
    
    
def cut_data(Xs, k):
    
    Xs_cut = torch.zeroes(Xs.shape)
    Xs_cut[:, k] = Xs[:, k].copy()
    
    return Xs_cut
    


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (args.lrdecay ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):

    if filename is not None:

        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')


def save_config(args, run_name = ''):

    file_name = run_name + '_config.json'
    config_file = args.outpath / file_name
    
    config_dict = {k:(str(v) if (isinstance(v, pathlib.PosixPath) or isinstance(v, pathlib.WindowsPath)) else v) for
                   k,v in args.__dict__.items()}    
    
    with open(config_file, 'w') as fn:
        json.dump(config_dict, fn, indent=2)
        

def cross_entropy_split(output, target):
    """ Compute label dependent and label independent parts of cross entropy loss
    output: Model output.
    target: Corresponding one-hot (!) target.
    """

    loss_dep = - (target * output).sum()
    loss_ind = torch.logsumexp(output, dim=-1).sum()
    
    return loss_dep, loss_ind 
    
    
def clear_gradients(model):
    for param in model.parameters():
        if param.requires_grad:
            param.grad = None
            