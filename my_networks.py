import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import MLP,GCNConv,GATConv, GATv2Conv, EdgeConv, DynamicEdgeConv, SAGEConv, GraphConv, TransformerConv, GINConv
from selection_layers import *
from my_selection_functions import direction_selection
from math import sqrt
        
def mean_pool(x,batch):
    return global_mean_pool(x,batch)

def max_pool(x,batch):
    return global_max_pool(x,batch)

class GCN(torch.nn.Module):
    def __init__(self,num_features,num_classes,hidden_size=64,num_layers=3):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        
        self.convs = torch.nn.ModuleList([
            GCNConv(hidden_size, hidden_size) for _ in range(1,num_layers)
        ])
        
        self.lin = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        
        for conv in self.convs:
            x = F.elu(x)
            x = conv(x,edge_index)
            
        x = self.lin(x)
        return x
    

class SelGCN(torch.nn.Module):
    def __init__(self,num_features,num_classes,hidden_size=64,selection_function=direction_selection,selection_count=1,num_layers=3):
        super().__init__()
        
        self.selection_function = selection_function
        
        self.conv1 = SelGCNConv(num_features, hidden_size, selection_count=selection_count)
        
        self.convs = torch.nn.ModuleList([
            SelGCNConv(hidden_size, hidden_size,selection_count=selection_count) for _ in range(1,num_layers)
        ])
        
        self.lin = torch.nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, sel_criteria, edge_index):
        
        #selections = get_selections(sel_criteria,edge_index,self.selection_function)
        selections = self.selection_function(sel_criteria,edge_index)
        
        x = self.conv1(x, edge_index, selections)
        
        for conv in self.convs:
            x = F.elu(x)
            x = conv(x,edge_index,selections)
            
        x = self.lin(x)
        return x
    

class SAGEConvNet(torch.nn.Module):
    def __init__(self,num_features,num_classes,hidden_size=64,num_layers=3):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_size)
        self.convs = torch.nn.ModuleList([
            SAGEConv(hidden_size, hidden_size) for _ in range(1,num_layers)
        ])
        self.lin = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        
        for conv in self.convs:
            x = F.elu(x)
            x = conv(x,edge_index)
            
        x = self.lin(x)
        return x
        

class SelSAGEConvNet(torch.nn.Module):
    def __init__(self,num_features,num_classes,hidden_size=64,selection_function=direction_selection,selection_count=1,num_layers=3):
        super().__init__()
        
        self.selection_function = selection_function
        
        self.conv1 = SelSAGEConv(num_features, hidden_size, selection_count=selection_count)
        self.convs = torch.nn.ModuleList([
            SelSAGEConv(hidden_size, hidden_size,selection_count=selection_count) for _ in range(1,num_layers)
        ])
        self.lin = torch.nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, sel_criteria, edge_index):
        
        selections = self.selection_function(sel_criteria,edge_index)
        
        x = self.conv1(x, edge_index, selections)
        
        for conv in self.convs:
            x = F.elu(x)
            x = conv(x,edge_index,selections)
        
        x = self.lin(x)
        return x

    
class GraphConvNet(torch.nn.Module):
    def __init__(self,num_features,num_classes,hidden_size=64,num_layers=3):
        super().__init__()
        self.conv1 = GraphConv(num_features, hidden_size)
        self.convs = torch.nn.ModuleList([
            GraphConv(hidden_size, hidden_size) for _ in range(1,num_layers)
        ])
        self.lin = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        
        for conv in self.convs:
            x = F.elu(x)
            x = conv(x,edge_index)
        
        x = self.lin(x)
        return x
        

class SelGraphConvNet(torch.nn.Module):
    def __init__(self,num_features,num_classes,hidden_size=64,selection_function=direction_selection,selection_count=1,num_layers=3):
        super().__init__()
        
        self.selection_function = selection_function
        
        self.conv1 = SelGraphConv(num_features, hidden_size, selection_count=selection_count)
        self.convs = torch.nn.ModuleList([
            SelGraphConv(hidden_size, hidden_size,selection_count=selection_count) for _ in range(1,num_layers)
        ])
        self.lin = torch.nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, sel_criteria, edge_index):
        
        selections = self.selection_function(sel_criteria,edge_index)
        
        x = self.conv1(x, edge_index, selections)
        
        for conv in self.convs:
            x = F.elu(x)
            x = conv(x,edge_index,selections)
        
        x = self.lin(x)
        return x


class GINConvNet(torch.nn.Module):
    def __init__(self,num_features,num_classes,hidden_size=64,num_layers=3,aggr="max"):
        super().__init__()

        self.conv1 = GINConv(MLP([num_features, hidden_size, hidden_size]),aggr=aggr) 
        self.convs = torch.nn.ModuleList([
            GINConv(MLP([hidden_size, hidden_size, hidden_size]),aggr=aggr)  for _ in range(1,num_layers)
        ])
        self.lin = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        
        for conv in self.convs:
            x = F.elu(x)
            x = conv(x,edge_index)
        
        x = self.lin(x)
        return x


class SelGINConvNet(torch.nn.Module):
    def __init__(self,num_features,num_classes,hidden_size=64,selection_function=direction_selection,selection_count=1,num_layers=3):
        super().__init__()
        
        self.selection_function = selection_function

        self.conv1 = SelGINConv(MLP([num_features, hidden_size, hidden_size]), num_features, selection_count=selection_count)
        self.convs = torch.nn.ModuleList([
            SelGINConv(MLP([hidden_size, hidden_size, hidden_size]), hidden_size, selection_count=selection_count) for _ in range(1,num_layers)
        ])
        self.lin = torch.nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, sel_criteria, edge_index):
        
        selections = self.selection_function(sel_criteria,edge_index)
        
        x = self.conv1(x, edge_index, selections)
        
        for conv in self.convs:
            x = F.elu(x)
            x = conv(x,edge_index,selections)
        
        x = self.lin(x)
        return x


class GAT(torch.nn.Module):
    def __init__(self,num_features,num_classes,hidden_size=256,num_layers=3,num_heads=4,version=2):
        super().__init__()
        
        if version == 2:
            self.conv1 = GATv2Conv(num_features, hidden_size, heads=num_heads)
            self.convs = torch.nn.ModuleList([
                GATv2Conv(num_heads * hidden_size, hidden_size, heads=num_heads) for _ in range(1,num_layers)
            ])
            self.lin = torch.nn.Linear(num_heads*hidden_size, num_classes)
            
        else:
            self.conv1 = GATConv(num_features, hidden_size, heads=num_heads)
            self.convs = torch.nn.ModuleList([
                GATConv(num_heads * hidden_size, hidden_size, heads=num_heads) for _ in range(1,num_layers)
            ])
            self.lin = torch.nn.Linear(num_heads*hidden_size, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        
        for conv in self.convs:
            x = F.elu(x)
            x = conv(x,edge_index)
        
        x = self.lin(x)
        return x


class TransformerConvNet(torch.nn.Module):
    def __init__(self,num_features,num_classes,hidden_size=256,num_layers=3,num_heads=4):
        super().__init__()
        self.conv1 = TransformerConv(num_features, hidden_size, heads=num_heads)
        self.convs = torch.nn.ModuleList([
            TransformerConv(num_heads * hidden_size, hidden_size, heads=num_heads) for _ in range(1,num_layers)
        ])
        self.lin = torch.nn.Linear(num_heads*hidden_size, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        
        for conv in self.convs:
            x = F.elu(x)
            x = conv(x,edge_index)
        
        x = self.lin(x)
        return x


class EdgeConvNet(torch.nn.Module):
    def __init__(self,num_features,num_classes,hidden_size=64,num_layers=3,aggr="max"):
        super().__init__()

        self.conv1 = EdgeConv(MLP([2*num_features, hidden_size, hidden_size]),aggr=aggr) 
        self.convs = torch.nn.ModuleList([
            EdgeConv(MLP([2*hidden_size, hidden_size, hidden_size]),aggr=aggr)  for _ in range(1,num_layers)
        ])
        self.lin = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        
        for conv in self.convs:
            x = F.elu(x)
            x = conv(x,edge_index)
        
        x = self.lin(x)
        return x
    

class DynamicEdgeConvNet(torch.nn.Module):
    def __init__(self,num_features,num_classes,hidden_size=64,num_layers=3,aggr="max"):
        super().__init__()

        self.conv1 = EdgeConv(MLP([2*num_features, hidden_size, hidden_size]),aggr=aggr) 
        self.convs = torch.nn.ModuleList([
            DynamicEdgeConv(MLP([2*hidden_size, hidden_size, hidden_size]),k=20,aggr=aggr)  for _ in range(1,num_layers)
        ])
        self.lin = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        
        for conv in self.convs:
            x = F.elu(x)
            x = conv(x,batch)
        
        x = self.lin(x)
        return x
    