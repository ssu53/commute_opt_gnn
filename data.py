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
        self.expander_edge_index = None

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

    def draw_expander(
        self, layout=partial(nx.spring_layout, seed=42), with_labels=True
    ):
        expander_edge_index_tuples = [
            (edge[0].item(), edge[1].item()) for edge in self.expander_edge_index.T
        ]
        g = nx.Graph(expander_edge_index_tuples)
        return nx.draw(g, pos=layout(g), with_labels=with_labels)


    def attach_expander_cayley(self):

        cg_gen = CayleyGraphGenerator(self.num_nodes)
        cg_gen.generate_cayley_graph()
        cg_gen.trim_graph()

        gexp = nx.relabel_nodes(
            cg_gen.G_trimmed, dict(zip(cg_gen.G_trimmed, range(self.num_nodes)))
        )
        gexp_edge_list = einops.rearrange(torch.tensor(list(gexp.edges)), "e n -> n e")
        
        self.expander_edge_index = gexp_edge_list



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

            vals = torch.rand((num_nodes * (num_nodes - 1) // 2,))
            vals = torch.where(
                vals < 1.0 - sparse_frac, vals / (1.0 - sparse_frac), 0.0
            )
            i, j = torch.triu_indices(num_nodes, num_nodes, offset=1)
            interact_strength = torch.zeros(num_nodes, num_nodes)
            interact_strength[i, j] = vals
            interact_strength.T[i, j] = vals

        self.interact_strength = interact_strength

    def _set_y(self, y):
        """
        graph regression target is the sum of
            2 c_ij tanh(x_i + x_j)
        where c_ij is the interaction strength between i and j
        """
        if y is None:
            x1 = einops.rearrange(self.x, "n d -> n 1 d")
            x2 = einops.rearrange(self.x, "n d -> 1 n d")
            y = torch.tanh(x1 + x2)
            assert y.shape == (self.num_nodes, self.num_nodes, self.dim_feat)
            y = torch.einsum("n m d, n m -> d", y, self.interact_strength)

        self.data.y = y

    def _set_edge_attr(self):
        """
        we are not concerned with edge attributes in this synthetic dataset
        """
        self.data.edge_attr = None
    
    def attach_expander(self, expander):
        if expander is None:
            pass
        elif expander == "cayley":
            self.attach_expander_cayley()
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
            interact_strength=self.interact_strength,
            id=self.id,
        )


class ExpMix(MyGraph):
    def __init__(self, id, data, x=None, y=None, dim_feat=16, expander=None, seed=42):
        super().__init__(id, data)

        torch.manual_seed(seed)
        random.seed(seed)

        self.dim_feat_synth = dim_feat
        self.feat_max = 1.0

        self._set_x(x)
        self._set_y(y)
        self.attach_expander(expander)

    def _set_x(self, x):
        """
        d-dimensional features
        there are <= d distinct pairs of interacting nodes
        each pair has random uniform features in a distinct dimension
        interaction appears in regression target as exp(x_i + x_j) term
        which is also the maximal mixing (Di Giovanni et al, 2023)
        """

        self.interact_strength = torch.zeros((self.num_nodes, self.num_nodes))

        if x is None:
            x = torch.zeros((self.num_nodes, self.dim_feat_synth))

            nodes = list(range(self.num_nodes))
            random.shuffle(nodes)

            ind_dims = list(range(self.dim_feat_synth))
            random.shuffle(ind_dims)

            num_interacting_pairs = random.randint(1, self.dim_feat_synth)

            for _ in range(num_interacting_pairs):
                if len(nodes) < 2:
                    break
                node1 = nodes.pop()
                node2 = nodes.pop()
                ind_dim = ind_dims.pop()
                x1 = random.uniform(0, self.feat_max)
                x2 = random.uniform(0, self.feat_max)

                x[node1, ind_dim] = x1
                x[node2, ind_dim] = x2
                self.interact_strength[node1, node2] = x1 + x2
                self.interact_strength[node2, node1] = x1 + x2

        self.data.x = x

        assert self.dim_feat_synth == self.dim_feat

    def _set_y(self, y):
        """
        graph regression target is
            \sum_d \sum_{i, j} exp(x_i + x_j)
        """

        if y is None:
            y = self.x.sum(dim=0)  # sum over node dimension
            y = torch.sum(torch.exp(y) - 1.0)

        self.data.y = y

    def _set_edge_attr(self):
        """
        we are not concerned with edge attributes in this synthetic dataset
        """
        self.data.edge_attr = None

    def attach_expander(self, expander):
        if expander is None:
            pass

        elif expander == "cayley":
            self.attach_expander_cayley()

        elif expander == "topk":
            K = 1

            gexp_edge_list = []

            interactions = [
                (i, j, self.interact_strength[i, j])
                for i in range(self.num_nodes)
                for j in range(i + 1, self.num_nodes)
            ]
            interactions_sorted = sorted(interactions, key=lambda x: x[2], reverse=True)

            for i, j, strength in interactions_sorted[:K]:
                gexp_edge_list.append((i, j))

            gexp_edge_list = einops.rearrange(
                torch.tensor(gexp_edge_list), "e n -> n e"
            )
            self.expander_edge_index = gexp_edge_list

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
            # in current form, this attribute is not compatible with graph batching
            # TODO: store instead as edge attribute. not urgent as model shouldn't access it anyways
            # interact_strength=self.interact_strength,
            expander_edge_index=self.expander_edge_index,
            id=self.id,
        )


class DoubleExp(MyGraph):
    def __init__(self, id, data, c1, c2, d, x=None, y=None, seed=42, expander=None):
        super().__init__(id, data)

        torch.manual_seed(seed)
        random.seed(seed)

        self.c1, self.c2, self.d = c1, c2, d

        self.distances = None

        self._set_x(x)
        self._set_y(y)
        self.attach_expander(expander)

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
        """

        if y is None:
            g = to_networkx(self.data, to_undirected=True)
            distances = dict(nx.all_pairs_shortest_path_length(g, cutoff=self.d))
            self.distances = distances

            y = 0

            for i in distances:
                for j, dist in distances[i].items():
                    if dist == 1:
                        y += self.c1 * torch.exp(self.data.x[i] + self.data.x[j])
                    elif dist == self.d:
                        y += self.c2 * torch.exp(self.data.x[i] + self.data.x[j])
                    else:
                        pass

            y = torch.tensor([y], dtype=torch.float)

        self.data.y = y

    def _set_edge_attr(self):
        """
        we are not concerned with edge attributes in this synthetic dataset
        """
        self.data.edge_attr = None

    def attach_expander(self, expander):
        if expander is None:
            pass

        elif expander == "cayley":
            self.attach_expander_cayley()

        elif expander == "interacting_pairs":
            
            distances = self.distances
            gexp_edge_list = []

            for i in distances:
                for j, dist in distances[i].items():
                    if (dist == 1) or (dist == self.d):
                        gexp_edge_list.append((i, j))
            
            gexp_edge_list = einops.rearrange(
                torch.tensor(gexp_edge_list), "e n -> n e"
            )
            self.expander_edge_index = gexp_edge_list

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
            expander_edge_index=self.expander_edge_index,
            id=self.id,
        )
