import torch

from torch_geometric.nn import MLP, GCNConv, SAGEConv, GraphConv, GINConv, EdgeConv, GATConv
from selection_layers import SelGCNConv, SelSAGEConv, SelGraphConv, SelGINConv

import torch.nn.functional as F

from my_networks_temporal import GCN, SelGCN, SAGEConvNet, SelSAGEConvNet, GraphConvNet, SelGraphConvNet, GATNet

class TemporalGNN(torch.nn.Module):
    r"""An implementation of the Attention Temporal Graph Convolutional Cell.
    For details see this paper: `"A3T-GCN: Attention Temporal Graph Convolutional
    Network for Traffic Forecasting." <https://arxiv.org/abs/2006.11583>`_

    Args:
        in_channels (int): Number of input features.
        hidden_size (int): Number of hidden features on each hidden layer
        periods (int): Number of time periods.
        improved (bool): Stronger self loops (default :obj:`False`).
        cached (bool): Caching the message weights (default :obj:`False`).
        add_self_loops (bool): Adding self-loops for smoothing (default :obj:`True`).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        periods: int,
        spatialGCN: str = "GCN",
        selection_count:int = 0,
        num_hidden_layers:int = 1,
        num_spatial_layers:int = 1
    ):
        super(TemporalGNN, self).__init__()

        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.periods = periods
        self.spatialGCN = spatialGCN
        self.selection_count = selection_count
        self.num_hidden_layers = num_hidden_layers
        self.num_spatial_layers = num_spatial_layers
        self._setup_layers()

    def _setup_layers(self):
        
        if self.spatialGCN[:3] == "Sel":
            self._base_tgcn = SelTGCN(
                in_channels=self.in_channels,
                out_channels=self.hidden_size,
                spatialGCN = self.spatialGCN,
                selection_count = self.selection_count,
                num_layers = self.num_spatial_layers
            )
            self.hidden_layers = torch.nn.ModuleList( 
                [SelTGCN(self.hidden_size,self.hidden_size,self.spatialGCN,self.selection_count) for _ in range(1,self.num_hidden_layers)])
        else:
            self._base_tgcn = TGCN(
                in_channels=self.in_channels,
                out_channels=self.hidden_size,
                spatialGCN = self.spatialGCN,
                num_layers = self.num_spatial_layers
            )
            self.hidden_layers = torch.nn.ModuleList(
                [TGCN(self.hidden_size,self.hidden_size,self.spatialGCN) for _ in range(1,self.num_hidden_layers)])

        self.linear = torch.nn.Linear(self.hidden_size,self.periods)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._attention = torch.nn.Parameter(torch.empty(self.periods, device=device))
        torch.nn.init.uniform_(self._attention)

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** (PyTorch Float Tensor): Node features for T time periods.
            * **edge_index** (PyTorch Long Tensor): Graph edge indices.
            * **edge_weight** (PyTorch Long Tensor, optional)*: Edge weight vector.
            * **H** (PyTorch Float Tensor, optional): Hidden state matrix for all nodes.

        Return types:
            * **H** (PyTorch Float Tensor): Hidden state matrix for all nodes.
        """
        
        H_accum = 0
        probs = torch.nn.functional.softmax(self._attention, dim=0)
        for period in range(self.periods):
            current = self._base_tgcn(X[:,:,:,period],edge_index,edge_weight,H)
            for net in self.hidden_layers:
                current = torch.tanh(current)
                current = net( current, edge_index, edge_weight, H)
            H_accum = H_accum + probs[period] * current
            
        H_accum = F.relu(H_accum)
        H_accum = self.linear(H_accum)
            
        #print(H_accum.shape)

        return H_accum
    

class TGCN(torch.nn.Module):
    r"""An implementation THAT SUPPORTS BATCHES of the Temporal Graph Convolutional Gated Recurrent Cell.
    For details see this paper: `"T-GCN: A Temporal Graph ConvolutionalNetwork for
    Traffic Prediction." <https://arxiv.org/abs/1811.05320>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
    """

    def __init__(self, in_channels: int, out_channels: int, spatialGCN: str, num_layers:int=1):
        super(TGCN, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatialGCN = spatialGCN
        self.num_layers = num_layers
        self._create_parameters_and_layers()

    def make_spatial_layer(self,name,in_channels,out_channels):
            
        if name == "GCN":
            conv_layer = GCN(in_channels,out_channels,self.num_layers)
        elif name == "SAGEConv":
            conv_layer = SAGEConvNet(in_channels,out_channels,self.num_layers)
        elif name == "GraphConv":
            conv_layer = GraphConvNet(in_channels,out_channels,self.num_layers)
        elif name == "GAT":
            conv_layer = GATNet(in_channels,out_channels,self.num_layers)
        else:
            raise ValueError("Spatial GCN name unknown")
            
        return conv_layer
        
    def _create_update_gate_parameters_and_layers(self):
        self.conv_z = self.make_spatial_layer(self.spatialGCN,self.in_channels,self.out_channels)
        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_r = self.make_spatial_layer(self.spatialGCN,self.in_channels,self.out_channels)
        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_h = self.make_spatial_layer(self.spatialGCN,self.in_channels,self.out_channels)
        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            # can infer batch_size from X.shape, because X is [B, N, F]
            H = torch.zeros(X.shape[0], X.shape[1], self.out_channels).to(X.device) #(b, 207, 32)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        #b,n,f = X.shape
        #Z = torch.cat([self.conv_z(X.view(b*n,f), edge_index, edge_weight).view(b,n,-1), H], axis=2) # (b, 207, 64)
        if edge_weight is None:
            Z = torch.cat([self.conv_z(X, edge_index), H], axis=2) # Fix for networks that cannot accept edge weight
        else:
            Z = torch.cat([self.conv_z(X, edge_index, edge_weight), H], axis=2) # (b, 207, 64)
        Z = self.linear_z(Z) # (b, 207, 32)
        Z = torch.sigmoid(Z)

        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        if edge_weight is None:
            R = torch.cat([self.conv_r(X, edge_index), H], axis=2) # Fix for networks that cannot accept edge weight
        else:
            R = torch.cat([self.conv_r(X, edge_index, edge_weight), H], axis=2) # (b, 207, 64)
        R = self.linear_r(R) # (b, 207, 32)
        R = torch.sigmoid(R)

        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        if edge_weight is None:
            H_tilde = torch.cat([self.conv_h(X, edge_index), H * R], axis=2) # Fix for networks that cannot accept edge weight
        else:
            H_tilde = torch.cat([self.conv_h(X, edge_index, edge_weight), H * R], axis=2) # (b, 207, 64)
        H_tilde = self.linear_h(H_tilde) # (b, 207, 32)
        H_tilde = torch.tanh(H_tilde)

        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde   # # (b, 207, 32)
        return H

    def forward(self,X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor = None,
                H: torch.FloatTensor = None ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde) # (b, 207, 32)
        return H
        
        
class SelTGCN(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, spatialGCN: str, selection_count: int, num_layers:int = 1):
        super(SelTGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatialGCN = spatialGCN
        self.selection_count = selection_count
        self.num_layers = num_layers

        self._create_parameters_and_layers()

    def make_spatial_layer(self,name,in_channels,out_channels,selection_count):

        if name == "SelGCN":
            conv_layer = SelGCN(in_channels,out_channels,selection_count=selection_count,num_layers=self.num_layers)
        elif name == "SelSAGEConv":
            conv_layer = SelSAGEConvNet(in_channels,out_channels,selection_count=selection_count,num_layers=self.num_layers)
        elif name == "SelGraphConv":
            conv_layer = SelGraphConvNet(in_channels,out_channels,selection_count=selection_count,num_layers=self.num_layers)
        else:
            raise ValueError("Spatial GCN name unknown")
            
        return conv_layer

    def _create_update_gate_parameters_and_layers(self):
        self.conv_z = self.make_spatial_layer(self.spatialGCN,self.in_channels,self.out_channels,selection_count=self.selection_count)
        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_r = self.make_spatial_layer(self.spatialGCN,self.in_channels,self.out_channels,selection_count=self.selection_count)
        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_h = self.make_spatial_layer(self.spatialGCN,self.in_channels,self.out_channels,selection_count=self.selection_count)
        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            # can infer batch_size from X.shape, because X is [B, N, F]
            H = torch.zeros(X.shape[0], X.shape[1], self.out_channels).to(X.device) #(b, 207, 32)
        return H

    def _calculate_update_gate(self, X, edge_index, selections, edge_weight, H):
        if edge_weight is None:
            Z = torch.cat([self.conv_z(X, edge_index, selections), H], axis=2) # (b, 207, 64)
        else:
            Z = torch.cat([self.conv_z(X, edge_index, selections, edge_weight), H], axis=2) # (b, 207, 64)
        Z = self.linear_z(Z) # (b, 207, 32)
        Z = torch.sigmoid(Z)

        return Z

    def _calculate_reset_gate(self, X, edge_index, selections, edge_weight, H):
        if edge_weight is None:
            R = torch.cat([self.conv_r(X, edge_index, selections), H], axis=2) # (b, 207, 64)
        else:
            R = torch.cat([self.conv_r(X, edge_index, selections, edge_weight), H], axis=2) # (b, 207, 64)
        R = self.linear_r(R) # (b, 207, 32)
        R = torch.sigmoid(R)

        return R

    def _calculate_candidate_state(self, X, edge_index, selections, edge_weight, H, R):
        if edge_weight is None:
            H_tilde = torch.cat([self.conv_h(X, edge_index, selections), H * R], axis=2) # (b, 207, 64)
        else:
            H_tilde = torch.cat([self.conv_h(X, edge_index, selections, edge_weight), H * R], axis=2) # (b, 207, 64)
        H_tilde = self.linear_h(H_tilde) # (b, 207, 32)
        H_tilde = torch.tanh(H_tilde)

        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde   # # (b, 207, 32)
        return H

    def forward(self,X: torch.FloatTensor, edge_index: torch.LongTensor, selections: torch.LongTensor, edge_weight: torch.FloatTensor = None,
                H: torch.FloatTensor = None ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, selections, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, selections, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, selections, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde) # (b, 207, 32)
        return H
    
