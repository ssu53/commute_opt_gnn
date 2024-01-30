import torch
import einops
from functools import partial

import networkx as nx
from torch_geometric.utils import to_networkx



class MyGraph:

    def __init__(self, id, data):
        self.id = id
        self.data = data

    @property
    def num_nodes(self):
        return self.data.x.shape[0]
    
    @property
    def num_edges(self):
        return self.data.edge_index.shape[1]
    
    @property
    def edge_index(self):
        return self.data.edge_index
    
    @property
    def dim_feat(self):
        return self.data.x.shape[1]

    @property
    def x(self):
        return self.data.x
    
    @property
    def y(self):
        return self.data.y
    
    @property
    def edge_attr(self):
        return self.data.edge_attr

    def draw(self, layout=partial(nx.spring_layout, seed=42)):
        g = to_networkx(self.data, to_undirected=True)
        return nx.draw(g, pos=layout(g))



class TanhMix(MyGraph):

    def __init__(self, id, data, x=None, y=None, interact_strength=None, seed=42):

        super().__init__(id, data)

        torch.manual_seed(seed)

        self._set_x(x)
        self._set_interact_strength(interact_strength)
        self._set_y(y)
        self._set_edge_attr()

    def _set_x(self, x):
        """
        set one-dimensional features, drawn uniformly from the unit interval
        """
        if x is None:
            num_nodes, dim_feat = self.num_nodes, 1
            x = torch.rand((num_nodes, dim_feat))

        self.data.x = x

    def _set_interact_strength(self, interact_strength, sparse_frac=0.75):
        """
        interaction strength is given by a symmetric matrix with zero diagonal
        the (i,j)'th element is the interaction between node i and node j
        sparse_frac of these elements are 0
        """
        if interact_strength is None:
            num_nodes = self.num_nodes
            
            vals = torch.rand((num_nodes*(num_nodes-1)//2,))
            vals = torch.where(vals < 1.0 - sparse_frac, vals / (1.0 - sparse_frac), 0.0)
            i, j = torch.triu_indices(num_nodes, num_nodes, offset=1)
            interact_strength = torch.zeros(num_nodes, num_nodes)
            interact_strength[i,j] = vals
            interact_strength.T[i,j] = vals

        self.interact_strength = interact_strength

    def _set_y(self, y):
        """
        graph regression target is the sum of 
            2 c_ij tanh(x_i + x_j)
        where c_ij is the interaction strength between i and j
        """
        if y is None:
            x1 = einops.rearrange(self.x, 'n d -> n 1 d')
            x2 = einops.rearrange(self.x, 'n d -> 1 n d')
            y = torch.tanh(x1 + x2)
            assert y.shape == (self.num_nodes, self.num_nodes, self.dim_feat)
            y = torch.einsum('n m d, n m -> d', y, self.interact_strength)

        self.data.y = y

    def _set_edge_attr(self):
        """
        we are not concerned with edge attributes in this synthetic dataset
        """
        self.data.edge_attr = None

