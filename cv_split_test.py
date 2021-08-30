#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 15:04:16 2021

@author: matthew

Source:https://www.machinecurve.com/index.php/2021/02/03/how-to-use-k-fold-cross-validation-with-pytorch/
"""
import numpy as np
from util import r2_score
from torch_geometric.data import DataLoader
import torch
from loadDataset import loadDataset
import wandb
from GIN import GIN0
from GCN import GCN
from PNA import PNA
from MLP import MLP
from EGC import EGC
from torch_geometric.utils import degree
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import KFold

from tqdm import trange

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


def generatePlot(argDataset, train_pred, train_actual, residual_train, test_pred, test_actual, residual_test, fold):
    sns.scatterplot(x=train_pred, y=train_actual)
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title('Predicted Vs. Actual - '+ argDataset + ' Train - ' + argModel + f' - Fold: {fold}')
    plt.show()
    
    sns.distplot(residual_train, fit=norm, kde=False)
    plt.xlabel('Errors')
    plt.title('Error Terms - '+ argDataset + ' Train - ' + argModel + f' - Fold: {fold}')
    plt.show()
    
    sns.scatterplot(x=test_pred, y=test_actual)
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title('Predicted Vs. Actual - '+ argDataset + ' Test - ' + argModel + f' - Fold: {fold}')
    plt.show()
    
    sns.distplot(residual_test, fit=norm, kde=False)
    plt.xlabel('Errors')
    plt.title('Error Terms - '+ argDataset + ' Test - ' + argModel + f' - Fold: {fold}')
    plt.show()

def reset_weights(m):
    '''
        Try resetting model weights to avoid
        weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()
            
def train(epoch, train_loader):
    model.train()
    
    current_loss = 0.0
    
    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        
        out = model(data)  # Perform a single forward pass.
        
        loss = criterion(out, data.y)

        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        
        current_loss += loss.item() # Get all loss from each batch
            
    current_loss = current_loss/len(train_loader)  # Average the loss across all batch in train
            
    return current_loss
            
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
# Config
# =============================================================================
argDataset      = "DG3x3"
argNormalize    = True
argBatchSize    = 64
argNumEpochs    = 1000
argLearningRate = 1e-2 #usually 1e-3
argSeedNum      = 24
argTorchSeed    = 24
argKFolds       = 5
argModel        = "EGC"
argWandb        = True
argProjectName  = argDataset
argScheduler    = "Plateau" #StepLR #Plateau

# For plotting
sns.set_theme()

# =============================================================================
# Set Seed
# =============================================================================
np.random.seed(argSeedNum)
torch.manual_seed(argTorchSeed)
torch.cuda.manual_seed(argTorchSeed)

dataset = loadDataset(argDataset, argNormalize)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=argKFolds, shuffle=True, random_state=argSeedNum)

# K-fold Cross Validation model evaluation
for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
    
    # WandB setup
    if argWandb == True:
        config = dict(
        epochs        = argNumEpochs,
        seed_num      = argSeedNum,
        torch_seed    = argTorchSeed,
        batch_size    = argBatchSize,
        learning_rate = argLearningRate,
        dataset       = argDataset,
        architecture  = "EGC_v2",
        total_fold    = argKFolds,
        Scheduler     = argScheduler)
        
        run = wandb.init(project=argProjectName, config=config)
    else:
        pass
    
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    # Sample elements randomly from a given list of ids, no replacement.
    train_idx = list(train_idx)
    test_idx = list(test_idx)
    
    train_fold = dataset[train_idx]
    test_fold = dataset[test_idx]
    
    # Define data loaders for training and testing data in this fold
    trainloader = DataLoader(train_fold, batch_size=argBatchSize)
    testloader = DataLoader(test_fold, batch_size=argBatchSize)
    
    # Insert train, val, test size in config for WandB
    if argWandb == True:
        wandb.config.update({"train_size": len(train_fold), "test_size": len(test_fold)})
    else:
        pass
    
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
        argHidden = 64
        model = GCN(argLayers, argHidden).to(device)
    elif argModel == "PNA":
        argLayers = 3
        argHidden = 50
        deg = torch.zeros(16, dtype=torch.long)
        for data in trainloader:
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
    
    if argWandb == True:
        wandb.config.update({"num_layers": argLayers, "hidden_dim": argHidden})
    else:
        pass
        
    optimizer = torch.optim.Adam(model.parameters(), lr=argLearningRate)
    criterion = torch.nn.MSELoss()
    
    model.apply(reset_weights)
    
    
    # Add learning rate decay, decay by 0.5 every 50 epochs
    if argScheduler == "Plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)
    elif argScheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    else:
        pass
    
    # Run the training loop for defined number of epochs
    for epoch in trange(0, argNumEpochs):

        # Set current loss value
        current_loss = 0.0
        
        # Training
        loss = train(epoch, trainloader)
        
        # Test
        with torch.no_grad():
            
            test_loss, test_outputs, test_labels = test(testloader)
            tr_loss, tr_outputs, tr_labels = test(trainloader)
            test_r2 = r2_score(test_labels, test_outputs)
            tr_r2 = r2_score(tr_labels, tr_outputs)
            
        print(f'\nFold:{fold}, \tTrain Loss: {loss:.4f}, \tTest Loss: {test_loss:.4f}')
        print(f'Train R2: {tr_r2:.4f}, \tTest R2: {test_r2:.4f}')
        
        # Record metrics to wandb
        if argWandb == True:
            wandb.log({"epoch": epoch,
                       "fold": fold,
                       "train loss": loss, 
                       "test loss": test_loss,
                       "train r2": tr_r2, 
                       "test r2": test_r2})
        else:
            pass
        
        # Initiate scheduler for learning rate decay
        if argScheduler == "Plateau":
            scheduler.step(test_r2)
        elif argScheduler == "StepLR":
            scheduler.step()
        else:
            pass
    
    train_pred = torch.detach(tr_outputs).numpy()
    train_actual = torch.detach(tr_labels).numpy()
    test_pred = torch.detach(test_outputs).numpy()
    test_actual = torch.detach(test_labels).numpy()
    
    residual_train = train_actual - train_pred
    residual_test = test_actual - test_pred
    
    std_residual_train = np.std(residual_train)
    std_residual_test = np.std(residual_test)
    
    if argWandb == True:
        wandb.log({"STD Residual Train": std_residual_train,
                   "STD Residual Test": std_residual_test})
    else:
        pass
    
    generatePlot(argDataset, train_pred, train_actual, residual_train, test_pred, test_actual, residual_test, fold)
    
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