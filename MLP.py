#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 17:48:58 2021

@author: matthew
"""
import torch
from torch.nn import Linear, Sequential, ReLU
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0))

class MLP(torch.nn.Module):
    def __init__(self, num_layers, hidden):
        super(MLP, self).__init__()
        
        self.fc1 = Sequential(Linear(2, hidden),
                              ReLU())
        
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(
                Sequential(Linear(hidden, hidden),
                            ReLU()))
            
        self.fc2 = torch.nn.Linear(hidden , 1)
        self.relu3 = torch.nn.ReLU()
        
        self.flatten = Flatten()
        
    def forward(self, data):
        x, batch = data.x, data.batch
        x = self.fc1(x)
        for conv in self.convs:
            x = conv(x)
        
        x = F.relu(self.fc2(x))    
        
        x = self.flatten(x)
        
        output = global_mean_pool(x, batch)
        
        return output
    
class Feedforward(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Feedforward, self).__init__()
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(2, int(self.hidden_size / 2))
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(int(self.hidden_size / 2), int(self.hidden_size / 4))  
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(int(self.hidden_size / 4) , 1)
        self.relu3 = torch.nn.ReLU()
        self.flatten = Flatten()
        
    def forward(self, data):
        x, batch = data.x, data.batch
        hidden = self.fc1(x)
        relu1 = self.relu(hidden)
        hidden2 = self.fc2(relu1)
        relu2 = self.relu2(hidden2)
        hidden3 = self.fc3(relu2)
        relu3 = self.relu3(hidden3)
        output = self.flatten(relu3)
        
        output = global_mean_pool(output, batch)  # [batch_size, hidden_channels]
        
        return output

# m1 = Feedforward(128)
# m2 = MLP(3, 64)