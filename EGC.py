#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 12:36:54 2021

@author: matthew
"""
import torch
import torch.nn.functional as F
from torch.nn import Sequential, ReLU, Linear, BatchNorm1d


from torch_geometric.nn import global_mean_pool, GCNConv
from conv.eg_conv import EGConv

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0))

class EGC(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, num_heads, num_bases, use_multi_aggregators = True):
        super(EGC, self).__init__()
        if use_multi_aggregators:
            aggregators = ['sum', 'mean', 'max', 'std']
        else:
            aggregators = ['symnorm']
        
        self.gcn = GCNConv(2, hidden_channels)

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                EGConv(hidden_channels, hidden_channels, aggregators,
                       num_heads, num_bases))
            self.norms.append(BatchNorm1d(hidden_channels))

        self.mlp = Sequential(
            Linear(hidden_channels, hidden_channels // 2, bias=False),
            BatchNorm1d(hidden_channels // 2),
            ReLU(inplace=True),
            Linear(hidden_channels // 2, hidden_channels // 4, bias=False),
            BatchNorm1d(hidden_channels // 4),
            ReLU(inplace=True),
            Linear(hidden_channels // 4, 1),
        )
        
        self.flatten = Flatten()

    def forward(self, data):
        x, edge_index,batch = data.x, data.edge_index,data.batch
        # adj_t = adj_t.set_value(None)  # EGConv works without any edge features
        
        x = self.gcn(x, edge_index)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index)
            h = norm(h)
            h = h.relu_()
            x = x + h

        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.mlp(x)
        x = self.flatten(x)

        return x
