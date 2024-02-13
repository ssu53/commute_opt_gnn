class TanhMix(MyGraph):
    def __init__(
        self,
        id,
        data,
        x=None,
        y=None,
        interact_strength=None,
        seed=42,
    ):
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

    def attach_rewirer(self, rewirer):
        if rewirer is None:
            pass
        elif rewirer == "cayley":
            self.attach_rewirer_cayley()
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
    def __init__(self, id, data, x=None, y=None, dim_feat=16, rewirer=None, seed=42):
        super().__init__(id, data)

        torch.manual_seed(seed)
        random.seed(seed)

        self.dim_feat_synth = dim_feat
        self.feat_max = 1.0

        self._set_x(x)
        self._set_y(y)
        self.attach_rewirer(rewirer)

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

    def attach_rewirer(self, rewirer):
        if rewirer is None:
            pass

        elif rewirer == "cayley":
            self.attach_rewirer_cayley()

        elif rewirer == "topk":
            K = 1

            rewire_edge_index = []

            interactions = [
                (i, j, self.interact_strength[i, j])
                for i in range(self.num_nodes)
                for j in range(i + 1, self.num_nodes)
            ]
            interactions_sorted = sorted(interactions, key=lambda x: x[2], reverse=True)

            for i, j, strength in interactions_sorted[:K]:
                rewire_edge_index.append((i, j))

            rewire_edge_index = einops.rearrange(
                torch.tensor(rewire_edge_index), "e n -> n e"
            )
            self.rewire_edge_index = rewire_edge_index

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
            rewire_edge_index=self.rewire_edge_index,
            id=self.id,
        )
