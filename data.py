import torch
import einops
from functools import partial
import random

import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

from generate_cayley_graph import CayleyGraphGenerator


class MyGraph:
    def __init__(self, id, data):
        self.id = id
        self.data = data
        self.rewire_edge_index = None

    @property
    def num_nodes(self):
        return self.data.num_nodes

    @property
    def num_edges(self):
        return self.data.num_edges

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

    def draw(self, layout=partial(nx.spring_layout, seed=42), with_labels=True):
        g = to_networkx(self.data, to_undirected=True)
        return nx.draw(g, pos=layout(g), with_labels=with_labels)

    def draw_rewire(self, layout=partial(nx.spring_layout, seed=42), with_labels=True):
        rewire_edge_index_tuples = [
            (edge[0].item(), edge[1].item()) for edge in self.rewire_edge_index.T
        ]
        g = nx.Graph(rewire_edge_index_tuples)
        return nx.draw(g, pos=layout(g), with_labels=with_labels)

    def attach_cayley(self):
        cg_gen = CayleyGraphGenerator(self.num_nodes)
        cg_gen.generate_cayley_graph()
        cg_gen.trim_graph()

        rewire_edge_index = nx.relabel_nodes(
            cg_gen.G_trimmed, dict(zip(cg_gen.G_trimmed, range(self.num_nodes)))
        )

        rewire_edge_index = einops.rearrange(
            torch.tensor(list(rewire_edge_index.edges)),
            "e n -> n e",
        )

        self.rewire_edge_index = rewire_edge_index

    def attach_fully_connected(self):
        rewire_edge_index = [
                (i, j)
                for i in range(self.num_nodes)
                for j in range(self.num_nodes)
                if i != j
            ]

        rewire_edge_index = einops.rearrange(
            torch.tensor(rewire_edge_index), "e n -> n e"
        )

        self.rewire_edge_index = rewire_edge_index



class ColourInteract(MyGraph):

    def __init__(self, id, data, c1, c2, num_colours, x=None, y=None, seed=42, rewirer=None):

        super().__init__(id, data)

        torch.manual_seed(seed)
        random.seed(seed)

        self.c1 = c1
        self.c2 = c2
        self.num_colours = num_colours

        self.values = None
        self.colours = None

        self._set_x(x)
        self._set_y(y)
        self.attach_rewirer(rewirer)
    
    
    def _set_values(self):
        """
        set one-dimensional "values", drawn uniformly from the unit interval
        """
        num_nodes, dim_feat = self.num_nodes, 1
        values = torch.rand((num_nodes, dim_feat))
        self.values = values
    
    def _set_colours(self):
        """
        set categorical "colours", uniformly from a set of self.size num_colours
        """
        colours = torch.randint(low=0, high=self.num_colours, size=(self.num_nodes,))
        self.colours = colours
    
    def _set_x(self, x):
        """
        set features as the concatenation of "values" and "colours"
        """
        if x is None:
            if self.values is None: self._set_values()
            if self.colours is None: self._set_colours()
            colours = torch.nn.functional.one_hot(self.colours, num_classes=self.num_colours)
            x = torch.hstack((self.values, colours))
            assert x.shape == (self.num_nodes, 1 + self.num_colours)
    
        self.data.x = x

    def _set_y(self, y):
        """
        Compute the regression target y for the graph as:
            \sum_{c \in colours} \sum_{i,j of colour c} c_1 exp(v_i + v_j)
            +
            \sum_{d} \sum_{i,j at distance d} c_2 2^{-d} exp(v_i + v_j)
        where v_i is the value associated with node i
        Excluding self-interactions (i.e. nodes separated by distance 0).
        Sum runs over distinct pairs only (i.e. (i,j) and (j,i) are not double-counted).
        """

        if y is None:
            g = to_networkx(self.data, to_undirected=True)
            distances = dict(nx.all_pairs_shortest_path_length(g))
            self.distances = distances

            y = 0.0

            for node_i in range(self.num_nodes):
                for node_j in range(node_i+1, self.num_nodes): # we choose to avoid self-interactions

                    interaction = torch.exp(self.values[node_i] + self.values[node_j])
                    
                    if self.colours[node_i].item() == self.colours[node_j].item():
                        y += self.c1 * interaction
                    
                    y += self.c2 * 2**(-distances[node_i][node_j]) * interaction
            
            y = torch.tensor([y], dtype=torch.float)
        
        self.data.y = y

    def _set_edge_attr(self):
        """
        we are not concerned with edge attributes in this synthetic dataset
        """
        self.data.edge_attr = None
    
    def attach_rewirer(self, rewirer):

        if rewirer is None:
            pass

        elif rewirer == "cayley":
            self.attach_cayley()
        
        elif rewirer == "fully_connected":
            self.attach_fullyconnected()
        
        else:
            raise NotImplementedError
    
    def to_torch_data(self):
        """
        casts to the usual torch Data object
        """
        return Data(
            x=self.x,
            y=self.y,
            edge_index=self.edge_index,
            rewire_edge_index=self.rewire_edge_index,
            id=self.id,
        )



class SalientDists(MyGraph):
    def __init__(self, id, data, c1, c2, c3, d, x=None, y=None, seed=42, rewirer=None):
        super().__init__(id, data)

        torch.manual_seed(seed)
        random.seed(seed)

        self.c1, self.c2, self.c3, self.d = c1, c2, c3, d

        self.distances = None

        self._set_x(x)
        self._set_y(y)
        self.attach_rewirer(rewirer)

    def _set_x(self, x):
        """
        set one-dimensional features, drawn uniformly from the unit interval
        """
        if x is None:
            num_nodes, dim_feat = self.num_nodes, 1
            x = torch.rand((num_nodes, dim_feat))

        self.data.x = x

    def _set_y(self, y):
        """
        Compute the regression target y for the graph as:
            \sum_{i,j at distance 1} c_1 exp(x_i + x_j)
            +
            \sum_{i,j at distance d} c_2 exp(x_i + x_j)
            +
            \sum_{i,j at distance other} c_3 exp(x_i + x_j)
        Excluding self-interactions (i.e. nodes separated by distance 0).
        Sum runs over distinct pairs only (i.e. (i,j) and (j,i) are not double-counted).
        """

        if y is None:
            g = to_networkx(self.data, to_undirected=True)
            distances = dict(nx.all_pairs_shortest_path_length(g))
            self.distances = distances

            y = 0

            for i in range(self.num_nodes):
                for j in range(i+1, self.num_nodes): # we choose to avoid self-interactions
                    dist = distances[i][j]
                    
                    if dist == 1:
                        y += self.c1 * torch.exp(self.data.x[i] + self.data.x[j])
                    elif dist == self.d:
                        y += self.c2 * torch.exp(self.data.x[i] + self.data.x[j])
                    else:
                        y += self.c3 * torch.exp(self.data.x[i] + self.data.x[j])

            y = torch.tensor([y], dtype=torch.float)

        self.data.y = y

    def _set_edge_attr(self):
        """
        we are not concerned with edge attributes in this synthetic dataset
        """
        self.data.edge_attr = None

    def attach_rewirer(self, rewirer):
        if rewirer is None:
            pass

        elif rewirer == "cayley":
            self.attach_cayley()
        
        elif rewirer == "interacting_pairs":
            rewire_edge_index = []

            for i in self.distances:
                for j, dist in self.distances[i].items():
                    if (dist == 1) or (dist == self.d):
                        rewire_edge_index.append((i, j))

            rewire_edge_index = einops.rearrange(
                torch.tensor(rewire_edge_index), "e n -> n e"
            )
            self.rewire_edge_index = rewire_edge_index
        
        elif rewirer == "fully_connected":
            self.attach_fully_connected()
        
        else:
            raise NotImplementedError

    def to_torch_data(self):
        """
        casts to the usual torch Data object
        """
        return Data(
            x=self.x,
            y=self.y,
            edge_index=self.edge_index,
            rewire_edge_index=self.rewire_edge_index,
            id=self.id,
        )
