#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 12:51:02 2021

@author: matthew
"""
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import os

def getData(json_file):
    # =============================================================================
    # Load data and transform data into appropriate shape
    # =============================================================================
    # Load json to pandas
    df = pd.read_json(json_file)
    
    # Take only important data
    df = df[['jssp_identification', 'optimal_time', 'jssp_instance']]
    
    y_time = df['optimal_time'][0]
    x_time = df['jssp_instance'][0]
    x_machine = df['jssp_instance'][1]
    
    new_df = pd.DataFrame([[x_time, x_machine, y_time]], columns=['process_time', 'machine', 'optimal_time'])
    
    return new_df

def getData3x3(json_file):
    df = pd.read_json(json_file)

    # Take only important data
    df = df[['jssp_identification', 'optimal_time', 'jssp_instance']]
    
    # Extract jssp_instance dict format to separate columns
    df = pd.concat([df.drop(['jssp_instance'], axis=1), df['jssp_instance'].apply(pd.Series)], axis=1)
    
    # =============================================================================
    # For job machine constraint
    # =============================================================================
    # Get machine number and processing time of each order
    mach = df[df[['0','1','2']]!=0].stack()
    
    # Make new column for combined data of (machine number, processing time)
    node_values = np.array([])
    for mach_index in range (len(mach)):
        time = mach[mach_index].astype(int)[0]
        machine = mach[mach_index].index.values.astype(int)[0]
        node_values = np.append(node_values, (machine, time))

    # =============================================================================
    # Get node features
    # =============================================================================
    node_features = np.reshape(node_values, (9,2))
    node_features = np.transpose(node_features) # row 0 is machine number, row 1 is time
    
    y_time = df['optimal_time'][0]
    x_time = node_features[1]
    x_machine = node_features[0]
    
    new_df = pd.DataFrame([[x_time, x_machine, y_time]], columns=['process_time', 'machine', 'optimal_time'])
    
    return new_df

def getNormVar(path_to_json, special_data=False):
    data_list = pd.DataFrame()
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
            
    # Extract all json to one DataFrame
    for json_file in tqdm(json_files):
        if special_data == True:
            data = getData3x3(path_to_json + json_file)
        else:
            data = getData(path_to_json + json_file)
        data_list = data_list.append(data)
        
    data_list = data_list.reset_index(drop=True)
    data_list['process_time'] = data_list['process_time'].apply(lambda x: np.array(x))
    
    return data_list

data_list = []
path_to_json = 'data/15x15x15/'
special_data = False #only True for 3x3x3
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
data_df = getNormVar(path_to_json, special_data)

big_time = np.array([])
big_machine = np.array([])
big_optimal = np.array([])
for i in range(len(data_df)):
    big_time = np.append(big_time, data_df['process_time'][i])
    big_machine = np.append(big_machine, data_df['machine'][i])
    big_optimal = np.append(big_optimal, data_df['optimal_time'][i])
    
    
import collections
c = collections.Counter(big_time)
od = collections.OrderedDict(sorted(c.items()))


import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")
sns.set_style("ticks")
sns.set_context("paper")
sns.histplot(data=big_time, discrete=True)
plt.xticks(np.arange(min(big_time), max(big_time)+1, 1.0))
plt.xlabel("Processing Time")
plt.title("Data Distribution - Processing Time - 6x6L2D")
plt.show()

sns.histplot(data=big_optimal, discrete=True)
plt.xticks(np.arange(min(big_optimal), max(big_optimal)+1, 3.0))
plt.xlabel("Optimal Time")
plt.title("Data Distribution - Optimal Time - 6x6L2D")
plt.show()

sns.histplot(data=big_machine, discrete=True)
plt.xticks(np.arange(min(big_machine), max(big_machine)+1, 1.0))
plt.xlabel("Machine Number")
plt.title("Data Distribution - Machine Number - 6x6L2D")
plt.show()