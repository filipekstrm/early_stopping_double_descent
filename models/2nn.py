## 2-Layer NN for CIFAR
## Based on https://github.com/zhangjh915/Two-Layer-Neural-Network/blob/master/neural_net.py

import torch.nn as nn

def make_nn(input_size=32*32*3, hidden_size=1000, num_classes=10):
    ''' Returns a 2-layer NN. '''
    return nn.Sequential(
        # Layer 0
        nn.Flatten()
        nn.Linear(input_size, hidden_size, bias=True)
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(),

        # Layer 1
        nn.Linear(hidden_size, num_classes, bias=True)
    )
