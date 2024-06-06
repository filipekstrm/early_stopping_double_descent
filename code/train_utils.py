import pathlib
import shutil
import json
import math
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision

import sys
sys.path.append('code/')
from linear_utils import default_p

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
    
    
    

def calculate_weight_mse(model, target, p=None, zero_eigs=0):

    output = extract_total_weights(model)
       
    output = output.squeeze() #numpy().squeeze(axis=-1) # TODO: Innan hade jag numpy här utan problem; hur?
    target = target.squeeze()
    
    assert output.shape == target.shape

    return calculate_weight_mse_raw(output, target, p, zero_eigs)
       
    
def calculate_weight_mse_raw(output, target, p=None, zero_eigs=0):
            
    assert output.shape == target.shape
    
    mse = (output - target)**2
    
    if p is not None:
        if zero_eigs > 0:
            mse = np.array((mse[:p].mean(), mse[p:-zero_eigs].mean(), mse[-zero_eigs:].mean()))
        else:
            mse = np.array((mse[:p].mean(), mse[p:].mean()))
        
    return mse 
    

def extract_total_weights(model):
    output = None
    for m in model:

        if type(m) == torch.nn.Linear:
            if output is None:
                output = m.weight.data.t()
            else:
                output = output @ m.weight.data.t()
        elif type(m) == ScalingLayer:
            output = m._theta()
        
    return output
    
def extract_weights(model):
    # Extract all model weights and return as 1D array
    weights = []
    
    for m in model:

        if type(m) == torch.nn.Linear:
            weights.append(m.weight.data.numpy().reshape(-1))
            
            if m.bias:
                weights.append(m.bias.data.numpy().reshape(-1))

        elif type(m) == ScalingLayer:
            output = weights.append(m._theta().numpy().reshape(-1))

    return np.concatenate(weights)
    
    
def get_weights_rank(model, num_layers):
    # Compute ranks of all layers but the last (it is a vector)
    ranks = []
    
    i = 0
    for m in model:

        if type(m) == torch.nn.Linear:
        
            if m.weight.data.isnan().any() or m.weight.data.isinf().any():
                ranks.append(float('nan'))
            
            else:
                ranks.append(torch.linalg.matrix_rank(m.weight.data))
            
            i += 1
            
        if i == (num_layers - 1):
            break

    return np.array(ranks)
    

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (args.lrdecay ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
            
            
            
# Initialisation

def kaiming_init(model, g_cpu, args):

    i = 0
    with torch.no_grad():
        for m in model:
            if type(m) == torch.nn.Linear:
                if i < (args.num_layers - 1): # NOTE: hade av misstag lika med här innan (i==0), så fem-lager-resultaten hade inte riktigt denna initialisering
                    torch.nn.init.kaiming_normal_(m.weight, a=math.sqrt(5)) #, generator=g_cpu)
                    m.weight.data = torch.mul(m.weight.data, args.scales[0])
                    print(m.weight.data.shape, args.scales[0])
                    
                elif i == (args.num_layers - 1): 
                    torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5)) #, generator=g_cpu)
                    m.weight.data = torch.mul(m.weight.data, args.scales[1])
                    print(m.weight.data.shape, args.scales[1])
                i += 1
            elif type(m) == ScalingLayer:
                torch.nn.init.kaiming_normal_(m.inner_weight, a=math.sqrt(5)) #, generator=g_cpu)
                m.inner_weight.data = torch.mul(m.inner_weight.data, args.scales[0])
                torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5)) #, generator=g_cpu)
                m.weight.data = torch.mul(m.weight.data, args.scales[1])

    return model
    
    
def fixed_init(model, args):
    i = 0
    with torch.no_grad():
        for m in model:
            if type(m) == torch.nn.Linear:
                if i < (args.num_layers - 1): 
                    m.weight.data = torch.ones(m.weight.data.shape) * args.scales[0]
                    print(m.weight.data.shape, args.scales[0])
                    
                    if (i == 0) and (args.small_eig_init_factor != 1.0):
                        p = count_largest_eigvals(args)
                        m.weight.data[:, p:] *= args.small_eig_init_factor
                        print(m.weight.data)
                        
                    
                elif i == (args.num_layers - 1): 
                    m.weight.data = torch.ones(m.weight.data.shape) * args.scales[1]
                    print(m.weight.data.shape, args.scales[1])
                i += 1
                
            elif type(m) == ScalingLayer:
                m.inner_weight.data = torch.ones(m.inner_weight.data.shape) * args.scales[0]
                m.weight.data = torch.ones(m.weight.data.shape) * args.scales[1]
                
    return model 
    
    
def rank_one_init(model, g_cpu, args):
    i = 0
    with torch.no_grad(): 
        p, q, u = 0, 0, 0
        for m in model:
            if type(m) == torch.nn.Linear:
                                
                if i == 0:
                
                    if args.fixed_weight_init:
                        q = torch.ones((m.weight.data.shape[1], 1)) * args.scales[0]
                    else:
                        q = torch.normal(mean=0, std=args.scales[0], size=(m.weight.data.shape[1], 1), generator=g_cpu)
                    
                    u = 1
                else:
                    q = p.clone()
                    
                    if i == 1: # u is then shared among the remaining layers
                        if args.fixed_weight_init:
                            u = torch.tensor(args.scales[1])
                        else:
                            u = torch.normal(mean=0, std=args.scales[1], size=(), generator=g_cpu)
                    
                p = torch.normal(mean=0, std=1, size=(m.weight.data.shape[0], 1), generator=g_cpu)
                p /= torch.norm(p, dim=0) # = (-)1 for last layer

                m.weight.data = u * torch.matmul(p, q.T)               

                i += 1
                
            elif type(m) == ScalingLayer:
                print("Low rank initialisation not implemented for ScalingLayer")
                
    return model 


def count_zero_eigvals(args):
    
    zero_eigs = 0 if args.dim <= args.samples else (args.dim - args.samples)
    
    return zero_eigs
    
    
def count_largest_eigvals(args):
    if args.kappa is None:
        p = None
    else:
        zero_eigs = count_zero_eigvals(args)
        p1 = args.eig_val_frac * args.dim if args.eig_val_frac is not None else default_p(args.dim, zero_eigs) # The number of eigenvalues equal to 1
        p = p1 if args.kappa <= 1.0 else (args.dim - zero_eigs - p1)
        
    return p
    
    
def get_w_min(Xs, ys, zero_eigs=0):
    if zero_eigs > 0:
        w_min = (np.transpose(Xs)@np.linalg.inv(Xs@np.transpose(Xs))@ys) #.squeeze()
    else:
        w_min = np.linalg.solve(np.transpose(Xs)@Xs, np.transpose(Xs)@ys)
    
    return w_min
    
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
        
        
class ScalingLayer(torch.nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_hidden):
        super().__init__()
        self.size_in, self.size_out = size_in, 1
        inner_weight = torch.Tensor(size_in, size_hidden)
        weight = torch.Tensor(self.size_out, size_in) 
        self.inner_weight = torch.nn.Parameter(weight)
        self.weight = torch.nn.Parameter(weight)  

        # initialize weights 
        torch.nn.init.kaiming_uniform_(self.inner_weight, a=np.sqrt(5)) 
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.inner_weight)
        
        torch.nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5)) 
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        
    def _theta(self):
        return (self.inner_weight * self.weight.sum(dim=-1, keepdims=True)).t()

    def forward(self, x):
        weight_mat = self._theta()
        return torch.mm(x, weight_mat)
    
