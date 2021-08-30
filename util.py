#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 14:07:34 2021

@author: matthew
"""
import numpy as np
import torch
import torch.utils.data as data_utils
import pandas as pd

def r2_score(target, prediction):
# =============================================================================
# Define R2 Score
# =============================================================================
    """Calculates the r2 score of the model
    
    Args-
        target- Actual values of the target variable
        prediction- Predicted values, calculated using the model
        
    Returns- 
        r2- r-squared score of the model
    """
    r2 = 1 - torch.sum((target-prediction)**2) / torch.sum((target-target.float().mean())**2)
    return r2

def toTensorDataset(json_file):
# =============================================================================
#     Converting json datafile to a tensor dataset
#     For mlp and ensemble boosting dataset
# =============================================================================
    df = pd.read_json(json_file)
    
    target = df['optimal_time'].values
    features = df['jobs_data'].values
    
    ex = np.array([])
    for index_feature in range(len(features)):
        # Make decimal rounding up to 4 digits
        features[index_feature] = np.round(features[index_feature], 4)
        
        # Extract ndarray object of numpy module to numpy array
        ex = np.append(ex, features[index_feature])
    
    # Reshape ex array to data size x node features size matrix
    input_size = len(features)
    input_shape = features[0].shape[0] * features[0].shape[1]
    ex = np.reshape(ex, (input_size, input_shape))
    
    # Convert Array to tensor
    target = torch.Tensor(target)
    ex = torch.Tensor(ex)

    # Create torch tensor dataset
    dataset = data_utils.TensorDataset(ex, target)
    
    return dataset
