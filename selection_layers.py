from typing import Callable, Optional, Union, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Embedding

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, SAGEConv, EdgeConv, GraphConv, GINConv
from torch_geometric.typing import Adj, OptTensor, OptPairTensor, PairOptTensor, PairTensor, NoneType, Size

from torch.nn.parameter import Parameter

from torch_geometric.nn.inits import reset, glorot, zeros
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax


try:
    from torch_cluster import knn
except ImportError:
    knn = None

from torch_geometric.nn.dense.linear import Linear
    
class SelectionConv(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int, selection_count = 9, **kwargs):
        super(SelectionConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.selection_count = selection_count
        #self.has_self_loops = has_self_loops

        self.weight = Parameter(torch.randn(self.selection_count,in_channels,out_channels,dtype=torch.float))
        torch.nn.init.uniform_(self.weight, a=-0.1, b=0.1)
        #torch.nn.init.normal_(self.weight)

        self.bias = Parameter(torch.randn(out_channels,dtype=torch.float))
        torch.nn.init.uniform_(self.bias, a=0.0, b=0.1)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, selections: Tensor) -> Tensor:
        """"""

        out = torch.zeros((x.shape[0],self.out_channels)).to(x.device)

        # Find the appropriate node for each selection by stepping through connecting edges
        for s in range(self.selection_count):
            cur_dir = torch.where(selections == s)[0]

            cur_source = edge_index[0,cur_dir]
            cur_target = edge_index[1,cur_dir]

            result = torch.matmul(x[cur_target], self.weight[s])

            # Adding with duplicate indices
            out.index_add_(0,cur_source,result)
            
        # Add bias if applicable
        out += self.bias

        return out

class SelGCNConv(GCNConv):
    
    def __init__(self, in_channels:int, out_channels:int, selection_count: int=9,
                 improved:bool = False, cached:bool = False, add_self_loops:bool=False, 
                 normalize:bool = False, bias: bool = True, **kwargs):
        
        super().__init__(in_channels, out_channels, improved, cached, add_self_loops, normalize, bias, **kwargs)
        
        self.selection_count = selection_count
        self.weight = Parameter(torch.randn(selection_count,out_channels,out_channels,dtype=torch.float))
        torch.nn.init.xavier_uniform_(self.weight)
        
    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, "weight"):
            torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x: Tensor, edge_index: Adj, selections: Tensor, edge_weight: OptTensor = None):
        self.cur_selections = selections
        return super().forward(x,edge_index,edge_weight)
    
    def message(self, x_j:Tensor, edge_weight:OptTensor):
        for s in range(self.selection_count):
            nodes = torch.where(self.cur_selections == s)[0]
            x_j[...,nodes,:] = torch.matmul(x_j[...,nodes,:], self.weight[s])
        
        return super().message(x_j,edge_weight)

    
class SelSAGEConv(SAGEConv):
    
    def __init__(self, in_channels:int, out_channels:int, selection_count: int=9,
                 aggr = "mean", normalize:bool = False, root_weight:bool = True,
                 project:bool = False, bias: bool = True, **kwargs):
        
        super().__init__(in_channels, out_channels, aggr, normalize, root_weight, project, bias, **kwargs)
        
        self.selection_count = selection_count
        self.weight = Parameter(torch.randn(selection_count,in_channels,in_channels,dtype=torch.float))
        torch.nn.init.xavier_uniform_(self.weight)
        
    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, "weight"):
            torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x: Tensor, edge_index: Adj, selections: Tensor):
        self.cur_selections = selections
        return super().forward(x,edge_index)
    
    def message(self, x_j:Tensor):
        for s in range(self.selection_count):
            nodes = torch.where(self.cur_selections == s)[0]
            x_j[...,nodes,:] = torch.matmul(x_j[...,nodes,:], self.weight[s])
        
        return x_j
        
        
class SelGraphConv(GraphConv):
    
    def __init__(self, in_channels:int, out_channels:int, selection_count: int=9,
                 aggr = "add", bias: bool = True, **kwargs):
        
        super().__init__(in_channels, out_channels, aggr, bias, **kwargs)
        
        self.selection_count = selection_count
        self.weight = Parameter(torch.randn(selection_count,in_channels,in_channels,dtype=torch.float))
        torch.nn.init.xavier_uniform_(self.weight)
        
    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, "weight"):
            torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x: Tensor, edge_index: Adj, selections: Tensor, edge_weight:OptTensor = None):
        self.cur_selections = selections
        return super().forward(x,edge_index,edge_weight)
    
    def message(self, x_j:Tensor, edge_weight: OptTensor):
        for s in range(self.selection_count):
            nodes = torch.where(self.cur_selections == s)[0]
            x_j[...,nodes,:] = torch.matmul(x_j[...,nodes,:], self.weight[s])
        
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class SelGINConv(GINConv):
    
    def __init__(self, nn: Callable, in_channels:int, selection_count: int=9,
                 eps: float = 0,train_eps: bool = False, **kwargs):
        
        super().__init__(nn, eps, train_eps, **kwargs)
        
        self.selection_count = selection_count
        self.weight = Parameter(torch.randn(selection_count,in_channels,in_channels,dtype=torch.float))
        torch.nn.init.xavier_uniform_(self.weight)
        
    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, "weight"):
            torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x: Tensor, edge_index: Adj, selections: Tensor):
        self.cur_selections = selections
        return super().forward(x,edge_index)
    
    def message(self, x_j:Tensor):
        for s in range(self.selection_count):
            nodes = torch.where(self.cur_selections == s)[0]
            x_j[...,nodes,:] = torch.matmul(x_j[...,nodes,:], self.weight[s])
        
        return x_j
