#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 14:44:52 2021

@author: matthew
"""
import numpy as np
from util import r2_score
from torch_geometric.data import DataLoader
import torch
from loadDataset import loadDataset
from GIN import GIN0
from GCN import GCN
from PNA import PNA
from MLP import MLP
from EGC import EGC
from GCN_edge import GCN_edge
from torch_geometric.utils import degree
from tqdm import trange
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from GPUtil import showUtilization as gputil_usage

# =============================================================================
# Clear cuda cache
# =============================================================================
with torch.no_grad():
    torch.cuda.empty_cache()

def getDataSplit(dataset):
    train_size = int(0.8 * len(dataset))
    total_test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, total_test_size], 
                                                                generator=torch.Generator().manual_seed(argSeedNum))
    
    test_size = int(total_test_size/2)
    val_size = test_size
    
    test_dataset, val_dataset= torch.utils.data.random_split(test_dataset, [test_size, val_size], 
                                                                generator=torch.Generator().manual_seed(argSeedNum))
    
    return train_dataset, val_dataset, test_dataset, train_size, val_size, test_size

# =============================================================================
# Config
# =============================================================================
argDataset      = "WG3x3"
argNormalize    = True
argBatchSize    = 64
argNumEpochs    = 500
argLearningRate = 1e-3
argSeedNum      = 24
argTorchSeed    = 24
argModel        = "GCN_edge"
argWandb        = False

# WandB setup
if argWandb == True:
    config = dict(
    epochs        = argNumEpochs,
    seed_num      = argSeedNum,
    torch_seed    = argTorchSeed,
    batch_size    = argBatchSize,
    learning_rate = argLearningRate,
    dataset       = argDataset,
    normalize     = argNormalize,
    architecture  = argModel)
    
    run = wandb.init(project="test", config=config)
else:
    pass
    

# =============================================================================
# Set Seed
# =============================================================================
np.random.seed(argSeedNum)
torch.manual_seed(argTorchSeed)
torch.cuda.manual_seed(argTorchSeed)

# =============================================================================
# Load Dataset
# =============================================================================
dataset = loadDataset(argDataset, argNormalize)

# =============================================================================
# Train test validation split
# =============================================================================
dataset = dataset.shuffle()

train_dataset, val_dataset, test_dataset, train_size, val_size, test_size = getDataSplit(dataset)

# Insert train, val, test size in config for WandB
if argWandb == True:
    wandb.config.update({"train_size": train_size, "val_size": val_size, "test_size": test_size})
else:
    pass

# =============================================================================
# Mini-batches
# 
# for val and test loader, the batch size is as big as possible to see
# the performance of the model for all data in val and test
# =============================================================================
train_loader = DataLoader(train_dataset, batch_size=argBatchSize, shuffle=True) # Shuffle at every epoch
val_loader = DataLoader(val_dataset, batch_size=test_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=val_size, shuffle=False)

# =============================================================================
# Initialize model
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if argModel == "GIN0":
    argLayers = 5
    argHidden = 128
    model = GIN0(argLayers, argHidden).to(device)
elif argModel == "GCN":
    argLayers = 3
    argHidden = 128
    model = GCN(argLayers, argHidden).to(device)
elif argModel == "PNA":
    argLayers = 3
    argHidden = 50
    deg = torch.zeros(16, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    model = PNA(deg).to(device)
elif argModel == "MLP":
    argLayers = 3
    argHidden = 128
    model = MLP(argLayers, argHidden).to(device)
elif argModel == "EGC":
    argLayers = 4
    argHidden = 128
    model = EGC(hidden_channels=argHidden, num_layers=argLayers, num_heads=4, num_bases=4).to(device)
elif argModel == "GCN_edge":
    argLayers = 3
    argHidden = 128
    model = GCN_edge(argLayers, argHidden).to(device)

if argWandb == True:
    wandb.config.update({"num_layers": argLayers, "hidden_dim": argHidden})
else:
    pass

optimizer = torch.optim.Adam(model.parameters(), lr=argLearningRate)
criterion = torch.nn.MSELoss()

# Scheduler
if argModel == "asdf":
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)
else:
    # Add learning rate decay, decay by 0.5 every 50 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)


def train(epoch):
    model.train()
    
    val_outputs = torch.tensor(()).to(device)
    val_labels = torch.tensor(()).to(device)
    
    current_loss = 0.0
    
    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        
        out = model(data)  # Perform a single forward pass.
        
        loss = criterion(out, data.y)

        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        
        current_loss += loss.item()
        
            
    with torch.no_grad():
        for val_data in val_loader:
            val_data = val_data.to(device)
            
            model.eval()
            
            val_out = model(val_data)
            val_loss = criterion(val_out, val_data.y)
            val_labels = torch.cat((val_labels, val_data.y))
            val_outputs = torch.cat((val_outputs, val_out))
            
    current_loss = current_loss/len(train_loader)  
            
    return val_loss, val_labels, val_outputs, current_loss

def test(loader):
    model.eval()
    
    outputs = torch.tensor(()).to(device)
    labels = torch.tensor(()).to(device)
    
    for data in loader:  # Iterate in batches over the training/test dataset.
       data = data.to(device)
       out = model(data)  
       loss = criterion(out, data.y)
       outputs = torch.cat((outputs, out))
       labels = torch.cat((labels, data.y))
    return loss.detach().cpu(), outputs.detach().cpu(), labels.detach().cpu()

# =============================================================================
# log with wandb
# =============================================================================
if argWandb == True:
    wandb.watch(model)
else:
    pass

# =============================================================================
# Main
for epoch in trange(1, argNumEpochs+1):
    # Training and get values from validation set
    val_loss, val_labels, val_outputs, train_loss = train(epoch)

    # Test
    test_loss, test_outputs, test_labels = test(test_loader)
    
    # Train
    tr_loss, tr_outputs, tr_labels = test(train_loader)
    
    
    val_r2 = r2_score(val_labels, val_outputs)
    test_r2 = r2_score(test_labels, test_outputs)
    tr_r2 = r2_score(tr_labels, tr_outputs)

    
    print(f'\nTrain Loss: {train_loss:.4f}, \tValidation Loss: {val_loss:.4f}, \tTest Loss: {test_loss:.4f}')
    print(f'Train R2: {tr_r2:.4f}, \tValidation R2: {val_r2:.4f}, \tTest R2: {test_r2:.4f}')
    
    # Record metrics to wandb
    if argWandb == True:
                wandb.log({"epoch": epoch, 
                           "train loss": train_loss, 
                           "val_loss": val_loss,
                           "test loss": test_loss,
                           "train r2": tr_r2, 
                           "val r2":val_r2 ,
                           "test r2": test_r2})
    else:
        pass
    
    if argModel == "asdf":
        scheduler.step(test_r2)
    else:
        scheduler.step()
    

    
# =============================================================================
# End wandb run
# =============================================================================
if argWandb == True:
    run.finish()
else:
    pass

# =============================================================================
# Clear cuda cache
# =============================================================================
with torch.no_grad():
    torch.cuda.empty_cache()
    
# # =============================================================================
# # Plotting
# # =============================================================================
# import matplotlib as plt
# import seaborn as sns
# from scipy.stats import norm

# train_pred = tr_outputs
# train_actual = tr_labels
# train_pred = torch.detach(train_pred).numpy()
# train_actual = torch.detach(train_actual).numpy()
# residual_train = train_actual - train_pred

# test_pred = test_outputs
# test_actual = test_labels
# test_pred = torch.detach(test_pred).numpy()
# test_actual = torch.detach(test_actual).numpy()
# residual_test = test_actual - test_pred

# print('std residual train: ', np.std(residual_train))
# print('std residual test: ', np.std(residual_test))

# sns.set_theme()

# sns.scatterplot(x=train_pred, y=train_actual)
# plt.xlabel('Predicted Values')
# plt.ylabel('Actual Values')
# plt.title('Predicted Vs. Actual - DG6x6 Train - EGConv')
# plt.show()

# sns.distplot(residual_train, fit=norm, kde=False)
# plt.xlabel('Errors')
# plt.title('Error Terms - DG6x6 Train - EGConv')
# plt.show()

# sns.scatterplot(x=test_pred, y=test_actual)
# plt.xlabel('Predicted Values')
# plt.ylabel('Actual Values')
# plt.title('Predicted Vs. Actual - DG6x6 Train - EGConv')
# plt.show()

# sns.distplot(residual_test, fit=norm, kde=False)
# plt.xlabel('Errors')
# plt.title('Error Terms - DG6x6 Test - EGConv')
# plt.show()