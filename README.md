# Improving Graph Networks Through Selection-based Convolution
This is the official implementation of the paper "Improving Graph Networks Through Selection-based Convolution" presented at WACV 2024.

[Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Hart_Improving_Graph_Networks_Through_Selection-Based_Convolution_WACV_2024_paper.pdf)

[Supplemental](https://openaccess.thecvf.com/content/WACV2024/supplemental/Hart_Improving_Graph_Networks_WACV_2024_supplemental.pdf)

This code has been tested with Python 3.10, PyTorch 2.0.1, PyTorch Geometric 2.4.0, and PyTorch Geometric Temporal 0.54. The results presented in the paper are saved as Tensorboard logs in `experiments.zip`

## Spatial Graph Networks
Selection-based graph convolution can improve the performance of spatial graph tasks. We provide code for running standard and selection-based graph network on spatial datasets. The following graph networks (PyG convention), datasets, and selection functions are built into our code:

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

Results can be viewed using Tensorboard in the specified log directory. The code can be modified to include additional graph networks, datasets, and selection functions by modifying the `my_networks.py`, `my_datasets.py`, and `my_selection_functions.py` respectively.

## Traffic Prediction
We implement various graph networks and selection-based variants for road traffic prediction using the PyTorch Geometric Temporal framework. By default, the code uses the A3TGCN temporal backbone for all graph networks. The following spatial aggregators, datasets, and selection functions are implemented:

**Spatial Aggregators**
- GCN
- SAGEConv
- GraphConv
- GAT
- Selection-based GCN (SelGCN)
- Selection-based SAGEConv (SelSAGEConv)
- Selection-based GraphConv (SelGraphConv)

**Datasets**
- METR-LA
- PEMS-BAY

**Selection Functions**
- Distance
- Direction (requires provided sensor location .csv files)
- Distance and Direction

To train a new graph network on one of the listed datasets, use the `train_traffic_network.py` code with your desired configuration. For example, to train a network with 3 Selection-based GCN spatial aggregator layers on the METR-LA dataset, run

```python train_traffic_network.py METRLA SelGCN --num_spatial 3 --selection_function distance --selection_count 4```

## Molecular Property Prediction
Selection-based networks can improve the performance of existing networks, or acheive similar performance with fewer parameters. This is demonstrated with our modification of DimeNet for the QM9 dataset. Our simple modifications to the network can be found in `my_dimenet.py`. The results from the paper can be regenerated for each target using the `train_QM9_network.py` code. For example, to train a network on the $\epsilon_{HOMO}$ (target #2) using selection-based DimeNet with 5 distance bins and 3 angle bins, run

```python train_QM9_network.py 02 SelDimeNet --selection_function distance_angle 5 3```

You can also use `SelDimeNetsmall` to train the smaller version of Selection-based DimeNet as described in the paper.
