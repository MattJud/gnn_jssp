#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 14:51:36 2021

@author: matthew
"""

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GCNConv, global_mean_pool

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0))

class GCN(torch.nn.Module):
    def __init__(self, num_layers, hidden):
        super(GCN, self).__init__()
             
        self.conv1 = GCNConv(2, hidden)
        
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GCNConv(hidden, hidden))

        self.lin = Linear(hidden, 1) #dataset.num_classes convert from float to int
        
        self.lin1 = Linear(hidden, hidden // 2)
        self.lin2 = Linear(hidden // 2, 1)
        
        self.flatten = Flatten()
        
    def forward(self, data):
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        
        for conv in self.convs:
            x = x.relu()
            x = conv(x,edge_index)
        
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin(x)
        
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        
        x = self.flatten(x)
        
        return x
    
# model = GCN(3, 64)