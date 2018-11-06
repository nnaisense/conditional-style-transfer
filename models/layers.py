import torch
import torch.distributions
import torch.nn as nn
from torch.nn import init
import functools
import numpy as np
from .utils import *
    
class PeerRegularizationLayerAtt(nn.Module):
    def __init__(self, att_dim, att_dropout_rate, K = 5, dist_metric = 'euclidean'):
        super(PeerRegularizationLayerAtt, self).__init__()
        
        self.att_dim = att_dim
        self.att_dropout_rate = att_dropout_rate
        self.K = K
        
        model = [torch.nn.Linear(in_features = att_dim, out_features = att_dim // 2),
                 torch.nn.BatchNorm1d(att_dim // 2),
                 torch.nn.ReLU(),
                 torch.nn.Linear(in_features = att_dim // 2, out_features = att_dim // 4),
                 torch.nn.BatchNorm1d(att_dim // 4),
                 torch.nn.ReLU()]
        self.transf_att = nn.Sequential(*model)
        self.conv_att1 = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = att_dim // 4)
        self.conv_att2 = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = att_dim // 4)
        self.att_dropout = torch.nn.Dropout(att_dropout_rate)
        
        self.pairwise_dist = PairwiseDistance(metric = dist_metric)
        
    def featmap_to_pixwise(self, fmap):
        fmap = torch.transpose(fmap, 1, 0)
        fmap = fmap.contiguous().view([fmap.shape[0], -1])
        fmap = torch.transpose(fmap, 1, 0)
        return fmap
    
    def pixwise_to_featmap(self, pwise, fmap_shape):
        pwise = torch.transpose(pwise, 1, 0)
        pwise = pwise.view([fmap_shape[1], fmap_shape[0], fmap_shape[2], fmap_shape[3]])
        pwise = torch.transpose(pwise, 1, 0)
        return pwise
    
    def recompose_style_features(self, inp_pwise, inp_pwise_style, peers_pwise, peers_pwise_style):
        # Compute k-NN of each pixel to get the adjacency information
        dist_matrix = self.pairwise_dist(inp_pwise, peers_pwise)
        # Get the top-K
        topk_vals, topk_idxs = torch.topk(dist_matrix, self.K, largest = False, sorted = True)
        
        inp_pixwise = self.transf_att(inp_pwise).unsqueeze(1)
        peers_pixwise = self.transf_att(peers_pwise).unsqueeze(1)
        
        att_1 = self.conv_att1(inp_pixwise)
        att_2 = self.conv_att2(peers_pixwise)
        att = att_1 + torch.transpose(att_2, 1, 0)
        att = att[:, :, 0]
        if self.att_dropout_rate > 0:
            att = self.att_dropout(att)
        
        knn_filter = torch.gt(dist_matrix, topk_vals[:, self.K-1:self.K]).type(torch.cuda.FloatTensor) * -1e9
        
        att = torch.nn.functional.softmax(torch.nn.functional.softplus(att) + knn_filter, dim = 1)
        
        out_pixwise_style = torch.matmul(att, peers_pwise_style)
        
        return out_pixwise_style
    
        
    def __call__(self, input):
        inp, peers = input
        inp_cont, inp_style = inp
        peers_cont, peers_style = peers
        
        inp_pixwise = self.featmap_to_pixwise(inp_cont)
        inp_pixwise_style = self.featmap_to_pixwise(inp_style)
        peers_pixwise = self.featmap_to_pixwise(peers_cont)
        peers_pixwise_style = self.featmap_to_pixwise(peers_style)
        
        out_pixwise_style = self.recompose_style_features(inp_pixwise, inp_pixwise_style, peers_pixwise, peers_pixwise_style)
                                         
        out_pixwise_style = self.pixwise_to_featmap(out_pixwise_style, inp_style.shape)
        output = torch.cat([inp_cont, out_pixwise_style], dim = 1)
        
        return output
    

class PeerRegularizationLayerAttBidir(nn.Module):
    def __init__(self, att_dim, att_dropout_rate, K = 5, dist_metric = 'euclidean'):
        super(PeerRegularizationLayerAttBidir, self).__init__()
        
        self.att_dim = att_dim
        self.att_dropout_rate = att_dropout_rate
        self.K = K
        
        model = [torch.nn.Linear(in_features = att_dim, out_features = att_dim // 2),
                 torch.nn.BatchNorm1d(att_dim // 2),
                 torch.nn.ReLU(),
                 torch.nn.Linear(in_features = att_dim // 2, out_features = att_dim // 4),
                 torch.nn.BatchNorm1d(att_dim // 4),
                 torch.nn.ReLU()]
        self.transf_att_cont = nn.Sequential(*model)
        self.conv_att1_cont = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = att_dim // 4)
        self.conv_att2_cont = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = att_dim // 4)
        self.att_dropout_cont = torch.nn.Dropout(att_dropout_rate)
        
        model = [torch.nn.Linear(in_features = att_dim * 2, out_features = att_dim),
                 torch.nn.BatchNorm1d(att_dim),
                 torch.nn.ReLU(),
                 torch.nn.Linear(in_features = att_dim, out_features = att_dim // 2),
                 torch.nn.BatchNorm1d(att_dim // 2),
                 torch.nn.ReLU()]
        self.transf_att_sty = nn.Sequential(*model)
        self.conv_att1_sty = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = att_dim // 2)
        self.conv_att2_sty = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = att_dim // 2)
        self.att_dropout_sty = torch.nn.Dropout(att_dropout_rate)
               
        self.pairwise_dist = PairwiseDistance(metric = dist_metric)
        
    def featmap_to_pixwise(self, fmap):
        fmap = torch.transpose(fmap, 1, 0)
        fmap = fmap.contiguous().view([fmap.shape[0], -1])
        fmap = torch.transpose(fmap, 1, 0)
        return fmap
    
    def pixwise_to_featmap(self, pwise, fmap_shape):
        pwise = torch.transpose(pwise, 1, 0)
        pwise = pwise.view([fmap_shape[1], fmap_shape[0], fmap_shape[2], fmap_shape[3]])
        pwise = torch.transpose(pwise, 1, 0)
        return pwise
    
    def recompose_style_features(self, inp_pwise, inp_pwise_style, peers_pwise, peers_pwise_style):
        # Compute k-NN of each pixel to get the adjacency information
        dist_matrix = self.pairwise_dist(inp_pwise, peers_pwise)
        # Get the top-K
        topk_vals, topk_idxs = torch.topk(dist_matrix, self.K, largest = False, sorted = True)
        
        inp_pwise = self.transf_att_cont(inp_pwise).unsqueeze(1)
        peers_pwise = self.transf_att_cont(peers_pwise).unsqueeze(1)
        
        att_1 = self.conv_att1_cont(inp_pwise)
        att_2 = self.conv_att2_cont(peers_pwise)
        att = att_1 + torch.transpose(att_2, 1, 0)
        att = att[:, :, 0]
        if self.att_dropout_rate > 0:
            att = self.att_dropout_cont(att)
        
        knn_filter = torch.gt(dist_matrix, topk_vals[:, self.K-1:self.K]).type(torch.cuda.FloatTensor) * -1e9
        
        att = torch.nn.functional.softmax(torch.nn.functional.softplus(att) + knn_filter, dim = 1)

        out_pixwise_style = torch.matmul(att, peers_pwise_style)
        
        return out_pixwise_style
    
    
    def recompose_content_features(self, inp_pwise, inp_pwise_style, peers_pwise, peers_pwise_style):
        # Compute k-NN of each pixel to get the adjacency information
        dist_matrix = self.pairwise_dist(inp_pwise_style, peers_pwise_style)
        # Get the top-K
        topk_vals, topk_idxs = torch.topk(dist_matrix, self.K, largest = False, sorted = True)
        
        inp_pwise_style = self.transf_att_sty(inp_pwise_style).unsqueeze(1)
        peers_pwise_style = self.transf_att_sty(peers_pwise_style).unsqueeze(1)
        
        att_1 = self.conv_att1_sty(inp_pwise_style)
        att_2 = self.conv_att2_sty(peers_pwise_style)
        att = att_1 + torch.transpose(att_2, 1, 0)
        att = att[:, :, 0]
        if self.att_dropout_rate > 0:
            att = self.att_dropout_sty(att)
        
        knn_filter = torch.gt(dist_matrix, topk_vals[:, self.K-1:self.K]).type(torch.cuda.FloatTensor) * -1e9
        
        att = torch.nn.functional.softmax(torch.nn.functional.softplus(att) + knn_filter, dim = 1)
        
        out_pixwise = torch.matmul(att, peers_pwise)
        
        return out_pixwise
    
        
    def __call__(self, input):
        if len(input) == 2:
            inp, peers = input
            cont_transf = True
            style_transf = True
        else:
            inp, peers, cont_transf, style_transf = input
        inp_cont, inp_style = inp
        peers_cont, peers_style = peers
        
        inp_pixwise = self.featmap_to_pixwise(inp_cont)
        inp_pixwise_style = self.featmap_to_pixwise(inp_style)
        peers_pixwise = self.featmap_to_pixwise(peers_cont)
        peers_pixwise_style = self.featmap_to_pixwise(peers_style)
        
        out_pixwise_style = self.recompose_style_features(inp_pixwise, inp_pixwise_style, peers_pixwise, peers_pixwise_style)
        out_pixwise = self.recompose_content_features(inp_pixwise, out_pixwise_style, peers_pixwise, peers_pixwise_style)
        
        if not cont_transf:
            out_pixwise = inp_pixwise
        if not style_transf:
            out_pixwise_style = inp_pixwise_style
            
        out_pixwise_style = self.pixwise_to_featmap(out_pixwise_style, inp_style.shape)
        out_pixwise = self.pixwise_to_featmap(out_pixwise, inp_cont.shape)
        
        output = torch.cat([out_pixwise, out_pixwise_style], dim = 1)
        
        return output
    
