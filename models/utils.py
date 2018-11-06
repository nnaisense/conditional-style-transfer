import torch
import torch.distributions
import torch.nn as nn
from torch.nn import init
import functools
import numpy as np

##################################################################################
# Distance computation
##################################################################################
class PairwiseDistance(nn.Module):
    def __init__(self, metric = 'euclidean'):
        super(PairwiseDistance, self).__init__()
        
        self.metric = metric
        
    def euclidean_dist(self, x, y):
        x_2 = torch.sum(torch.pow(x, 2), dim = 1, keepdim = True)
        y_2 = torch.sum(torch.pow(y, 2), dim = 1, keepdim = True)
        dists = x_2 - 2 * torch.matmul(x, torch.transpose(y, 0, 1)) + torch.transpose(y_2, 0, 1)
        return dists
    
    def cosine_dist(self, x, y):
        x_norm = x / torch.norm(x, p = 2, dim = 1, keepdim = True)
        y_norm = y / torch.norm(y, p = 2, dim = 1, keepdim = True)
        dist_matrix = torch.mm(x_norm, y_norm.t())
        return 1.0 - dist_matrix
    
    def __call__(self, input1, input2):
        if self.metric == 'euclidean':
            dists = self.euclidean_dist(input1, input2)
        elif self.metric == 'cosine':
            dists = self.cosine_dist(input1, input2)
        else:
            raise NotImplementedError('Distance metric [%s] is not recognized' % self.metric)
            
        return dists
    