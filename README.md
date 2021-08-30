# gnn_jssp
Master's Thesis - Matthew Judijanto - RWTH Aachen 2021

Title: **Graph Neural Networks for Compact Representation for Job Shop Scheduling Problems: A Comparative Benchmark**

# Dependecies

1. [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric)
2. PyTorch 1.7.0

# Dataset

Datasets of this thesis are generated based on:
1. Normal datasets: https://www.scitepress.org/Link.aspx?doi=10.5220/0010202405890597
2. L2D datasets: https://arxiv.org/abs/2010.12367

Note: Dataset for problem above the size of 10x10 are removed in PyG form due to GitHub 50mb file size limit.

# Models

- Baseline:
-- MLP `MLP.py`
-- Ensemble Boosting `RunEnsembleBoosting.py`

- GNN Models:
-- GCN `GCN.py`
-- GIN-0 `GIN.py`
-- PNA `PNA.py`
-- EGC `EGC.py`

# To Run

To run and train the models, there are 2 ways to do this. For a simple single run, you can use `RunModel.py`. For running a K-Fold Cross Validation use `cv_split_test.py`

# Others

`DataExtractor.py` `DataExtractorNorm.py` `DisjunctiveGraph.py` `DisjunctiveGraph_v2.py` are used to convert the raw dataset from JSON to PyG graph data.

`loadDataset.py` is to load the PyG graph datasets to be runned on the model
