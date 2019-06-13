import numpy as np
from copy import copy
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
    
def to_torch(np_array, longtensor=False):
    if not longtensor:
        return torch.from_numpy(np_array.astype(np.float32))
    if longtensor:
        return torch.from_numpy(np_array.astype(np.long))

def to_numpy(src): 
    if type(src) == np.ndarray:
        return src
    else:
        x = src
    return x.detach().cpu().numpy()

def to_binary(tensor):
    tensor = (tensor >= 0.5).float().cuda()
    return tensor

def roll(x, n, dim=0):  
    return torch.cat((x[-n:], x[:-n]), dim=dim)