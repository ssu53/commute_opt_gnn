class GINVanilla(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int,
        eps: float = 0.0,
        bias: bool = True,
    ):
        """
        Args
            in_channels: dimension of input features
            hidden_channels: dimension of hidden layers
            num_layers: number of layers
            out_channels: dimension of output
            eps: (fixed, non-trainable) epsilon for GIN layers
            bias: whether to use bias for linear layers
        """

        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.eps = eps
        self.bias = bias
        self.pool = global_add_pool

        self.lin_layers = []
        if num_layers > 1:
            self.lin_layers += [
                GINConv(nn.Linear(in_channels, hidden_channels, bias), eps)
            ]
            self.lin_layers += [
                GINConv(nn.Linear(hidden_channels, hidden_channels, bias), eps)
                for _ in range(num_layers - 2)
            ]
            self.lin_layers += [
                GINConv(nn.Linear(hidden_channels, out_channels, bias), eps)
            ]
        else:
            self.lin_layers += [
                GINConv(nn.Linear(in_channels, out_channels, bias), eps)
            ]
        self.lin_layers = nn.ModuleList(self.lin_layers)

    def forward(self, data):
        """
        Args
            x: input
            edge_index: input graph edge index of shape (2, num_edges)
        """

        x = data.x
        edge_index = data.edge_index

        for i in range(self.num_layers):
            x = self.lin_layers[i](x, edge_index)
            if i != self.num_layers - 1:
                x = nn.functional.relu(x)

        x = self.pool(x, data.batch)  # (num_nodes, d) -> (batch_size, d)
        x = x.view(-1)  # (batch_size, d) -> (batch_size,)

        return x


class GINExpander(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int,
        eps: float = 0.0,
        bias: bool = True,
    ):
        """
        Args
            in_channels: dimension of input features
            hidden_channels: dimension of hidden layers
            num_layers: number of layers
            out_channels: dimension of output
            eps: (fixed, non-trainable) epsilon for GIN layers
            bias: whether to use bias for linear layers
        """

        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.eps = eps
        self.bias = bias
        self.pool = global_add_pool

        # GIN message passing with linear layer
        self.lin_layers = []
        if num_layers > 1:
            self.lin_layers += [
                GINConv(nn.Linear(in_channels, hidden_channels, bias), eps)
            ]
            self.lin_layers += [
                GINConv(nn.Linear(hidden_channels, hidden_channels, bias), eps)
                for _ in range(num_layers - 2)
            ]
            self.lin_layers += [
                GINConv(nn.Linear(hidden_channels, out_channels, bias), eps)
            ]
        else:
            self.lin_layers += [
                GINConv(nn.Linear(in_channels, out_channels, bias), eps)
            ]
        self.lin_layers = nn.ModuleList(self.lin_layers)

        # GIN message passing without trainable parameters
        self.id_layers = [GINConv(nn=nn.Identity(), eps=eps) for _ in range(num_layers)]
        self.id_layers = nn.ModuleList(self.id_layers)

    def forward(self, data):
        """
        Args
            x: input
            edge_index: input graph edge index of shape (2, num_edges)
            expander_edge_index: expander graph edge index of shape (2, num_expander_edges)
        """

        x = data.x
        edge_index = data.edge_index
        expander_edge_index = data.expander_edge_index

        for i in range(self.num_layers):
            x = self.lin_layers[i](x, edge_index)
            x = nn.functional.relu(x)
            x = self.id_layers[i](x, expander_edge_index)

        x = self.pool(x, data.batch)  # (num_nodes, d) -> (batch_size, d)
        x = x.view(-1)  # (batch_size, d) -> (batch_size,)

        return x
