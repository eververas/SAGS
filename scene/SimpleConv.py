from typing import Tuple, Union

from torch import Tensor

from torch_scatter import scatter_std, scatter_mean

from torch_geometric.nn.conv import MessagePassing
from torch import nn 
import torch.nn.functional as F

from torch_geometric.typing import Adj, OptTensor, PairTensor

import inspect


class DiffAggregator(MessagePassing):

    def __init__(self,  **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: PairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        out = (x_i - x_j) 
        return out if edge_weight is None else out * edge_weight.view(-1, 1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}()')


class Normalization(MessagePassing):

    def __init__(self,  **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)
        mu  = scatter_mean(x[0][edge_index[1]] , edge_index[0], dim=0)
        std = scatter_std (x[0][edge_index[1]] , edge_index[0], dim=0)

        # propagate_type: (x: PairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, mu=mu, std=std, edge_weight=edge_weight)

        return out

    def message(self, x_i: Tensor, mu_j: Tensor, std_j: Tensor, edge_weight: OptTensor) -> Tensor:
        out = (x_i - mu_j) / (std_j + 1e-15)
        return out if edge_weight is None else out * edge_weight.view(-1, 1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}()')
    

class SimpleGCN(MessagePassing):

    def __init__(self,  input_channels , hidden_size,  out_channels , **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.fc1 =  nn.Sequential(
                            nn.Linear(input_channels, hidden_size),
                            nn.ReLU(True),
                            nn.Linear(hidden_size, hidden_size),
                        )
        
        self.fc2 = nn.Linear(input_channels, hidden_size)
        
        self.fc3 = nn.Sequential(
                            nn.Linear(hidden_size, out_channels),
                            nn.ReLU(True)
                        )

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        # if isinstance(x, Tensor):
        #     x = (x, x)

        features = self.fc1(x)
        # propagate_type: (x: PairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, feats=features,  edge_weight=edge_weight)

        out = F.relu(out + self.fc2(x))
        out = self.fc3(out)
        return out

    def message(self, feats_i: Tensor, edge_weight: OptTensor) -> Tensor:
        out = feats_i 
        return out if edge_weight is None else out * edge_weight.view(-1, 1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}()')    
