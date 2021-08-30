#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 16:32:28 2021

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
          
    big_time = np.array([])
    big_machine = np.array([])
    for i in range(len(data_list)):
        big_time = np.append(big_time, data_list['process_time'][i])
        big_machine = np.append(big_machine, data_list['machine'][i])
    
    # time_var = big_time.var()
    # time_mean = big_time.mean()
    
    time_max = big_time.max()
    time_min = big_time.min()
    
    machine_max = big_machine.max()
    machine_min = big_machine.min()
    
    return time_max, time_min, machine_max, machine_min

def MinMaxNorm(x_input, max_val, min_val):
    # Normalize feature between 0.1 - 1
    # z_out = 0.1 + ((1-0.1)*(x_input - time_min))/(time_max - time_min)
    min_val = 0
    z_out = (x_input - min_val)/(max_val-min_val)
    return z_out


def DataExtractNorm(path_to_json, json_file, time_max, time_min, machine_max, machine_min, special_data=False):
    if special_data == True: # For loading 3x3x3 dataset
        df = getData3x3(path_to_json + json_file)

        y_time = df['optimal_time'][0]
        x_time = df['process_time'][0]
        x_machine = df['machine'][0]
        
        x_time = np.reshape(x_time, (3,3))
        x_machine = np.reshape(x_machine, (3,3))
        
        # Make new DataFrame
        new_df = pd.DataFrame(columns=['jobs', 'machine', 'time'])
        
        # Insert values to new DataFrame
        index = 0
        for i in range(len(x_time)):
            for j in range(len(x_time[i])):
                new_df.loc[index] = [i] + [x_machine[i][j]] + [x_time[i][j]] #values = index, jobs, machine, time
                index += 1 #increase index number for next iter
                
    else:
        # =============================================================================
        # Load data and transform data into appropriate shape
        # =============================================================================
        # Load json to pandas
        df = pd.read_json(path_to_json + json_file)
        
        # Take only important data
        df = df[['jssp_identification', 'optimal_time', 'jssp_instance']]
        
        y_time = df['optimal_time'][0]
        x_time = df['jssp_instance'][0]
        x_machine = df['jssp_instance'][1]
        
        # Make new DataFrame
        new_df = pd.DataFrame(columns=['jobs', 'machine', 'time'])
        
        # Insert values to new DataFrame
        index = 0
        for i in range(len(x_time)):
            for j in range(len(x_time[i])):
                new_df.loc[index] = [i] + [x_machine[i][j]] + [x_time[i][j]] #values = index, jobs, machine, time
                index += 1 #increase index number for next iter
        
        # Convert 'object' dtype to 'int'
        new_df = new_df.astype(str).astype(int)
    
    # =============================================================================
    # Normalize time input value
    # =============================================================================
    new_df['time'] = new_df['time'].apply(lambda x: MinMaxNorm(x, time_max, time_min))
    new_df['machine'] = new_df['machine'].apply(lambda x: MinMaxNorm(x, machine_max, machine_min))
    
    # =============================================================================
    # Node features
    # =============================================================================
    node_values = np.array([])
    for data_index in range(len(new_df.index)):
        input_time = new_df['time'][data_index]
        input_machine = new_df['machine'][data_index]
        node_values = np.append(node_values, (input_machine, input_time))
    
    n_nodes = int(len(node_values)/2)   
    node_features = np.reshape(node_values, (n_nodes,2)) 
        
    # Add node features for S and T nodes, 2 for S, 3 for T
    node_features = np.append(node_features, [[2,0],[3,0]], axis=0)
    
    # =============================================================================
    # Order constraint edges
    # =============================================================================
    # Get source and target nodes
    source_nodes = np.array([])
    target_nodes = np.array([])
    
    # Create directed edges from node S and T
    j_source_nodes = np.array([])
    j_target_nodes = np.array([])
    
    S_node = len(new_df)
    T_node = S_node + 1     # Next node after S_node
    
    groupped = new_df.groupby('jobs') # Group by jobs
    for job, group in groupped:
        # print(job)
        # print(group)
        
        # From S_node to first node (order = 0) in the job
        j_source_nodes = np.append(j_source_nodes, S_node)
        j_target_nodes = np.append(j_target_nodes, group.index.values.astype(int).min())
        
        # From end node (order = end) in the job to T_node 
        j_source_nodes = np.append(j_source_nodes, group.index.values.astype(int).max())
        j_target_nodes = np.append(j_target_nodes, T_node)
        
        # Create edges constraint for order in job
        for group_index in range(len(group.index.values.astype(int))-1):
            source_nodes = np.append(source_nodes, group.index.values.astype(int)[group_index]) 
            target_nodes = np.append(target_nodes, group.index.values.astype(int)[group_index+1]) # Target node is the node next to the souce node
    
    # =============================================================================
    # Machine constraint edges
    # =============================================================================
    m_source_nodes = np.array([])
    m_target_nodes = np.array([])
    
    # Group nodes by machine and create undirected graph (2-way directed graph) between nodes
    groupped_machine = new_df.groupby('machine')
    for n_machine, group_machine in groupped_machine:
        # print(group_machine)
        for m_group_index in range(len(group_machine.index.values.astype(int))):
            all_index = group_machine.index.values.astype(int)
            machine_index_source = all_index[m_group_index]
            machine_index_target = np.delete(all_index, m_group_index)
            # print(machine_index_source)
            # print(machine_index_target)
            
            for m_index in range(len(machine_index_target)):
                m_source_nodes = np.append(m_source_nodes, machine_index_source)
                m_target_nodes = np.append(m_target_nodes, machine_index_target[m_index])      
            
    # =============================================================================
    # Combine order sequence edges and machine constraint edges
    # =============================================================================
    complete_source_nodes = np.append(source_nodes, m_source_nodes)
    complete_target_nodes = np.append(target_nodes, m_target_nodes)
    
    # Add S and T nodes edges
    complete_source_nodes = np.append(complete_source_nodes, j_source_nodes)
    complete_target_nodes = np.append(complete_target_nodes, j_target_nodes)
    
    # Convert dtype from numpy float to int before inserting to PyG dataset
    complete_source_nodes = complete_source_nodes.astype(int)
    complete_target_nodes = complete_target_nodes.astype(int)
    
    # =============================================================================
    # Compile PyG dataset from extracted raw data
    # =============================================================================
    node_features = torch.FloatTensor(node_features) # Change from Long to Float, not sure if this is correct
    edge_index = torch.tensor([complete_source_nodes, complete_target_nodes], dtype=torch.long)
    x = node_features
    y = torch.FloatTensor([y_time]) #Don't forget to insert y value in array
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data

class DG3x3(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DG3x3, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['../input/DG3x3.pt'] #Has to create 'input' folder if hasn't been created yet, otherwise Errno 2,no such directory will be thrown

    def download(self):
        pass
    
    def process(self):
        
        data_list = []
        path_to_json = 'data/3x3x3/'
        json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
        time_max, time_min, machine_max, machine_min = getNormVar(path_to_json, special_data=True)
        
        # Extract all json to one DataFrame
        for json_file in tqdm(json_files):
            data = DataExtractNorm(path_to_json, json_file, time_max, time_min, machine_max, machine_min, special_data=True)
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class DG6x6(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DG6x6, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['../input/DG6x6.pt'] #Has to create 'input' folder if hasn't been created yet, otherwise Errno 2,no such directory will be thrown

    def download(self):
        pass
    
    def process(self):
        
        data_list = []
        path_to_json = 'data/6x6x6/'
        json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
        time_max, time_min, machine_max, machine_min = getNormVar(path_to_json)
        
        # Extract all json to one DataFrame
        for json_file in tqdm(json_files):
            data = DataExtractNorm(path_to_json, json_file, time_max, time_min, machine_max, machine_min)
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
class DG6x6L2D(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DG6x6L2D, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['../input/DG6x6L2D.pt'] #Has to create 'input' folder if hasn't been created yet, otherwise Errno 2,no such directory will be thrown

    def download(self):
        pass
    
    def process(self):
        
        data_list = []
        path_to_json = 'data/6x6x6_rs400_l2d/'
        json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
        time_max, time_min, machine_max, machine_min = getNormVar(path_to_json)
        
        # Extract all json to one DataFrame
        for json_file in tqdm(json_files):
            data = DataExtractNorm(path_to_json, json_file, time_max, time_min, machine_max, machine_min)
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class DG8x8(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DG8x8, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['../input/DG8x8.pt'] #Has to create 'input' folder if hasn't been created yet, otherwise Errno 2,no such directory will be thrown

    def download(self):
        pass
    
    def process(self):
        
        data_list = []
        path_to_json = 'data/8x8x8/'
        json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
        time_max, time_min, machine_max, machine_min = getNormVar(path_to_json)
        
        # Extract all json to one DataFrame
        for json_file in tqdm(json_files):
            data = DataExtractNorm(path_to_json, json_file, time_max, time_min, machine_max, machine_min)
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
class DG10x10(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DG10x10, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['../input/DG10x10.pt'] #Has to create 'input' folder if hasn't been created yet, otherwise Errno 2,no such directory will be thrown

    def download(self):
        pass
    
    def process(self):
        
        data_list = []
        path_to_json = 'data/10x10x10/'
        json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
        time_max, time_min, machine_max, machine_min = getNormVar(path_to_json)
        
        # Extract all json to one DataFrame
        for json_file in tqdm(json_files):
            data = DataExtractNorm(path_to_json, json_file, time_max, time_min, machine_max, machine_min)
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class DG15x15(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DG15x15, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['../input/DG15x15.pt'] #Has to create 'input' folder if hasn't been created yet, otherwise Errno 2,no such directory will be thrown

    def download(self):
        pass
    
    def process(self):
        
        data_list = []
        path_to_json = 'data/15x15x15/'
        json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
        time_max, time_min, machine_max, machine_min = getNormVar(path_to_json)
        
        # Extract all json to one DataFrame
        for json_file in tqdm(json_files):
            data = DataExtractNorm(path_to_json, json_file, time_max, time_min, machine_max, machine_min)
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
class DG15x15L2D(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DG15x15L2D, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['../input/DG15x15L2D.pt'] #Has to create 'input' folder if hasn't been created yet, otherwise Errno 2,no such directory will be thrown

    def download(self):
        pass
    
    def process(self):
        
        data_list = []
        path_to_json = 'data/15x15x15_rs400_l2d/'
        json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
        time_max, time_min, machine_max, machine_min = getNormVar(path_to_json)
        
        # Extract all json to one DataFrame
        for json_file in tqdm(json_files):
            data = DataExtractNorm(path_to_json, json_file, time_max, time_min, machine_max, machine_min)
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

dataset = DG3x3(root='')
