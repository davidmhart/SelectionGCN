import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, GATConv
from selection_layers import *
from math import sqrt

class GCN(torch.nn.Module):
    def __init__(self,num_features,hidden_size=64,num_layers=1):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        
        self.convs = torch.nn.ModuleList([
            GCNConv(hidden_size, hidden_size) for _ in range(1,num_layers)
        ])

    def forward(self, x, edge_index, edge_weights=None):
        x = self.conv1(x, edge_index, edge_weights)
        
        for conv in self.convs:
            x = F.elu(x)
            x = conv(x,edge_index,edge_weights)
            
        return x
    
class SelGCN(torch.nn.Module):
    def __init__(self,num_features,hidden_size=64,selection_count=1,num_layers=1):
        super().__init__()
        
        self.conv1 = SelGCNConv(num_features, hidden_size, selection_count=selection_count)
        
        self.convs = torch.nn.ModuleList([
            SelGCNConv(hidden_size, hidden_size,selection_count=selection_count) for _ in range(1,num_layers)
        ])     
        
    def forward(self, x, edge_index, selections, edge_weights=None):
        
        x = self.conv1(x, edge_index, selections, edge_weights)
        
        for conv in self.convs:
            x = F.elu(x)
            x = conv(x,edge_index,selections,edge_weights)
            
        return x

    
class SAGEConvNet(torch.nn.Module):
    def __init__(self,num_features,hidden_size=64,num_layers=1):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_size)
        self.convs = torch.nn.ModuleList([
            SAGEConv(hidden_size, hidden_size) for _ in range(1,num_layers)
        ])

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        
        for conv in self.convs:
            x = F.elu(x)
            x = conv(x,edge_index)
            
        return x
        
class SelSAGEConvNet(torch.nn.Module):
    def __init__(self,num_features,hidden_size=64,selection_count=1,num_layers=1):
        super().__init__()
        
        self.conv1 = SelSAGEConv(num_features, hidden_size, selection_count=selection_count)
        self.convs = torch.nn.ModuleList([
            SelSAGEConv(hidden_size, hidden_size,selection_count=selection_count) for _ in range(1,num_layers)
        ])
        
    def forward(self, x, edge_index, selections):
        
        x = self.conv1(x, edge_index, selections)
        
        for conv in self.convs:
            x = F.elu(x)
            x = conv(x,edge_index,selections)
        
        return x

    
class GraphConvNet(torch.nn.Module):
    def __init__(self,num_features,hidden_size=64,num_layers=1):
        super().__init__()
        self.conv1 = GraphConv(num_features, hidden_size)
        self.convs = torch.nn.ModuleList([
            GraphConv(hidden_size, hidden_size) for _ in range(1,num_layers)
        ])

    def forward(self, x, edge_index, edge_weights=None):
        x = self.conv1(x, edge_index, edge_weights)
        
        for conv in self.convs:
            x = F.elu(x)
            x = conv(x,edge_index,edge_weights)
        
        return x
        
class SelGraphConvNet(torch.nn.Module):
    def __init__(self,num_features,hidden_size=64,selection_count=1,num_layers=1):
        super().__init__()
        
        self.conv1 = SelGraphConv(num_features, hidden_size, selection_count=selection_count)
        self.convs = torch.nn.ModuleList([
            SelGraphConv(hidden_size, hidden_size,selection_count=selection_count) for _ in range(1,num_layers)
        ])
        
    def forward(self, x, edge_index, selections, edge_weights=None):
        
        x = self.conv1(x, edge_index, selections, edge_weights)
        
        for conv in self.convs:
            x = F.elu(x)
            x = conv(x,edge_index,selections,edge_weights)
        
        return x


class GATNetave(torch.nn.Module):
    def __init__(self,num_features,hidden_size=64,num_layers=1):
        super().__init__()
        self.out_size = hidden_size
        self.conv1 = GATConv(num_features, hidden_size, heads=4, concat=False)
        
        self.convs = torch.nn.ModuleList([
            GATConv(hidden_size, hidden_size, heads=4, concat=False) for _ in range(1,num_layers)
        ])

    def forward(self, x, edge_index, edge_weights=None):
        out_x = torch.zeros((x.shape[0],x.shape[1],self.out_size),device=x.device)
        for i in range(x.shape[0]):
            out_x[i] = self.conv1(x[i], edge_index, edge_weights)
        
        x = out_x

        for conv in self.convs:
            x = F.elu(x)
            out_x = torch.zeros((x.shape[0],x.shape[1],self.out_size),device=x.device)
            for i in range(x.shape[0]):
                out_x[i] = conv(x[i],edge_index,edge_weights)

            x = out_x

        return x

class GATNet(torch.nn.Module):
    def __init__(self,num_features,hidden_size=64,num_layers=1):
        super().__init__()
        self.out_size = 4*hidden_size
        self.conv1 = GATConv(num_features, hidden_size, heads=4, concat=True)

        self.convs = torch.nn.ModuleList([
            GATConv(4*hidden_size, hidden_size, heads=4, concat=True) for _ in range(1,num_layers)
        ])

        self.final_linear = torch.nn.Linear(4*hidden_size,hidden_size)

    def forward(self, x, edge_index, edge_weights=None):
        out_x = torch.zeros((x.shape[0],x.shape[1],self.out_size),device=x.device)
        for i in range(x.shape[0]):
            out_x[i] = self.conv1(x[i], edge_index, edge_weights)
        
        x = out_x

        for conv in self.convs:
            x = F.elu(x)
            out_x = torch.zeros((x.shape[0],x.shape[1],self.out_size),device=x.device)
            for i in range(x.shape[0]):
                out_x[i] = conv(x[i],edge_index,edge_weights)

            x = out_x

        x = self.final_linear(x)

        return x
    
