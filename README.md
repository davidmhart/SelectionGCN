# Improving Graph Networks Through Selection-based Convolution
This is the official implementation of the paper presented at WACV 2024.

[Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Hart_Improving_Graph_Networks_Through_Selection-Based_Convolution_WACV_2024_paper.pdf)

[Supplemental](https://openaccess.thecvf.com/content/WACV2024/supplemental/Hart_Improving_Graph_Networks_WACV_2024_supplemental.pdf)

This code has been tested with Python 3.10, PyTorch 2.0.1, PyTorch Geometric 2.4.0, and PyTorch Geometric Temporal 0.54

## Spatial Graph Networks
Selection-based graph convolution can improve the performance of spatial graph datasets. We provide code for running standard and selection-based graph network on spatial datasets. The following graph networks (PyG convention), datasets, and selection functions are built-in to our code:

**Graph Networks**
- GCN
- SAGEConv
- GraphConv
- GINConv
- GAT
- TransformerConv
- EdgeConv
- Selection-based GCN (SelGCN)
- Selection-based SAGEConv (SelSAGEConv)
- Selection-based GraphConv (SelGraphConv)
- Selection-based GINConv (SelGINConv)

**Datasets**
- MNIST Superpixels
- CoMA (must be downloaded externally)

**Selection Functions**
- Direction
- Distance
- Direction and Distance

To train a new graph network on one of the listed datasets, use the `train_spatial_network.py` code with your desired configuration (see the argument parser for all available options). For example, to train a basic GCN network on the label and position data of MNIST Superpixels, run

```python train_spatial_network.py MNIST GCN x+pos```

When using a selection-based network, you can specify the selection features and selection function. For example, to train a Selection-based GCN which selects on 6 directions, run

```python train_spatial_network.py MNIST SelGCN x --selection_criteria pos --selection_function direction --selection_count 7```

*Note*: Selection count is 7 to include a 0th direction for nodes that connect to themselves.
