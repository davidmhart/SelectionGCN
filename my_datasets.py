import os.path as osp
from sys import platform

from torch_geometric.datasets import MNISTSuperpixels, CoMA, QM9
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.transforms import FaceToEdge, RandomNodeSplit

import torch
import numpy as np

if platform == "win32" or platform == "win64":
    data_dir = "D:\Datasets\PytorchGeometric"
else:
    data_dir = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')

neighborhood = [25,10]
    
def find_max_shared(x,edge_index,max_check=0):
    from torch_geometric.utils import subgraph    

    if max_check > 0:
        subset = torch.arange(max_check).to(x.device)
        edge_index, _ = subgraph(subset,edge_index,relabel_nodes=True)
        x=x[:max_check]

    bool_x = x.bool()

    source = bool_x[edge_index[0]]
    target = bool_x[edge_index[1]]

    shared = torch.sum(torch.logical_and(source,target),dim=1)

    return torch.amax(shared)

        
class QM9_dataset():
    def __init__(self,target_num=None,path=None,batch_size=1,num_workers=0):
        if path is None:
            path = osp.join(data_dir, 'QM9')
        self.path = path
        dataset = QM9(path)
        
        # DimeNet uses the atomization energy for targets U0, U, H, and G, i.e.:
        # 7 -> 12, 8 -> 13, 9 -> 14, 10 -> 15
        idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 11])
        dataset.data.y = dataset.data.y[:, idx]
        
        if target_num is not None:
            dataset.data.y = dataset.data.y[:, target_num]
        
        #split_size = 10000
        #train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset)-2*split_size,split_size,split_size])
        
        # Use the same random seed as the official DimeNet` implementation.
        random_state = np.random.RandomState(seed=42)
        perm = torch.from_numpy(random_state.permutation(np.arange(130831)))
        train_idx = perm[:110000]
        val_idx = perm[110000:120000]
        test_idx = perm[120000:]
        
        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]
        test_dataset = dataset[test_idx]
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        if target_num is None:
            self.num_classes = 12
        else:
            self.num_classes = 1
        #self.criteria = torch.nn.BCEWithLogitsLoss()
        self.criteria = torch.nn.MSELoss()
        self.task_type = "graph multi-task"
        self.global_pool = True
        self.num_selections = 1
        self.selection_count = None
        
class MNISTsup_dataset():
    def __init__(self,path=None,batch_size=1,num_workers=0):
        if path is None:
            path = osp.join(data_dir, 'MNISTsup')
        self.path = path
        dataset = MNISTSuperpixels(path, train=True)
        print(len(dataset))
        split_size = int(.8*len(dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [split_size, len(dataset)-split_size])
        test_dataset = MNISTSuperpixels(path, train=False)
        print(len(test_dataset))
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        self.num_features = 1
        self.num_classes = 10
        self.criteria = torch.nn.CrossEntropyLoss()
        self.task_type = "graph label"
        self.global_pool = True
        self.selection_function = "VectorList2D"
        self.selection_count = 9
        self.num_selections = 1
        
class CoMA_dataset():
    def __init__(self,path=None,batch_size=1,num_workers=0):
        if path is None:
            path = osp.join(data_dir, 'CoMA')
        self.path = path
        dataset = CoMA(path, train=True, pre_transform=FaceToEdge())
        print(len(dataset))
        split_size = int(.8*len(dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [split_size, len(dataset)-split_size])
        test_dataset = CoMA(path, train=False, pre_transform=FaceToEdge())
        print(len(test_dataset))
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        self.num_features = 3
        self.num_classes = 12
        self.criteria = torch.nn.CrossEntropyLoss()
        self.task_type = "graph label"
        self.global_pool = True
        self.selection_function = "VectorList3D"
        self.selection_count = 27
        self.num_selections = 1
