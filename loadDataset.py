#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 17:45:57 2021

@author: matthew
"""

# =============================================================================
# Load dataset
# =============================================================================
from DataExtractor import JobShopDataset3x3, JobShopDataset6x6, JobShopDataset6x6L2D, JobShopDataset8x8, JobShopDataset10x10, JobShopDataset15x15, JobShopDataset15x15L2D
from DataExtractorNorm import JobShopDataset3x3Norm ,JobShopDataset6x6Norm, JobShopDataset6x6L2DNorm, JobShopDataset8x8Norm, JobShopDataset10x10Norm, JobShopDataset15x15Norm, JobShopDataset15x15L2DNorm
from DisjunctiveGraph import DisjunctiveGraph3x3, DisjunctiveGraph6x6, DisjunctiveGraph6x6L2D, DisjunctiveGraph8x8, DisjunctiveGraph10x10, DisjunctiveGraph15x15, DisjunctiveGraph15x15L2D
from DisjunctiveGraph_v2 import DG3x3
from WeightedGraph import WeightedGraph3x3


def loadDataset(dataset_name, normalization=True):
    if dataset_name == "3x3":
        if normalization is True:
            dataset = JobShopDataset3x3Norm(root='')
        else:
            dataset = JobShopDataset3x3(root='')
    elif dataset_name == "6x6":
        if normalization is True:
            dataset = JobShopDataset6x6Norm(root='')
        else:
            dataset = JobShopDataset6x6(root='')
            
    elif dataset_name == "6x6L2D":
        if normalization is True:
            dataset = JobShopDataset6x6L2DNorm(root='')
        else:
            dataset = JobShopDataset6x6L2D(root='')
            
    elif dataset_name == "8x8":
        if normalization is True:
            dataset = JobShopDataset8x8Norm(root='')
        else:
            dataset = JobShopDataset8x8(root='')
        
    elif dataset_name == "10x10":
        if normalization is True:
            dataset = JobShopDataset10x10Norm(root='')
        else:
            dataset = JobShopDataset10x10(root='')
        
    elif dataset_name == "15x15":
        if normalization is True:
            dataset = JobShopDataset15x15Norm(root='')
        else:
            dataset = JobShopDataset15x15(root='')
        
    elif dataset_name == "15x15L2D":
        if normalization is True:
            dataset = JobShopDataset15x15L2DNorm(root='')
        else:
            dataset = JobShopDataset15x15L2D(root='')
               
# =============================================================================
#     Disjunctive Graph
# =============================================================================
    elif dataset_name == "DG3x3":
        dataset = DisjunctiveGraph3x3(root='')
    
    elif dataset_name == "DG6x6":
        dataset = DisjunctiveGraph6x6(root='')
        
    elif dataset_name == "DG6x6L2D":
        dataset = DisjunctiveGraph6x6L2D(root='')
        
    elif dataset_name == "DG8x8":
        dataset = DisjunctiveGraph8x8(root='')
        
    elif dataset_name == "DG10x10":
        dataset = DisjunctiveGraph10x10(root='')
        
    elif dataset_name == "DG15x15":
        dataset = DisjunctiveGraph15x15(root='')
        
    elif dataset_name == "DG15x15L2D":
        dataset = DisjunctiveGraph15x15L2D(root='')
        
    elif dataset_name == "v2_DG3x3":
        dataset = DG3x3(root='')
    
    elif dataset_name == "WG3x3":
        dataset = WeightedGraph3x3(root='')
    
    return dataset