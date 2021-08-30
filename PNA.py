#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 18:00:44 2021

@author: matthew
"""

import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU, Linear, Dropout
from torch_geometric.nn import PNAConv, BatchNorm, global_add_pool
from torch_geometric.nn import GCNConv

from GPUtil import showUtilization as gputil_usage

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0))

class PNA(torch.nn.Module):
    def __init__(self, deg):
        super(PNA, self).__init__()

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        
        self.conv1 = GCNConv(2, 50)

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(3):
            conv = PNAConv(in_channels=50, out_channels=50,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           towers=1, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(50))

        self.mlp = Sequential(#Linear(75, 50), 
                              #ReLU(), 
                              Linear(50, 25), 
                              ReLU(), 
                              Dropout(p=0.2), #originally, there is no dropout here
                              Linear(25, 1))
        
        self.flatten = Flatten()

    def forward(self, data):
        # gputil_usage()
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
            
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))

        x = global_add_pool(x, batch)
        x = self.mlp(x)
        x = self.flatten(x)
        return x