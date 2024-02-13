import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_add_pool


class MyGINConv(MessagePassing):
    def __init__(self, dim_emb):
        """
        GINConv with built-in MLP (linear -> batch norm -> relu -> linear)
        Args
            dim_emb: embedding dimension
        """

        super().__init__(aggr="add")

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_emb, 2 * dim_emb),
            torch.nn.BatchNorm1d(2 * dim_emb),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * dim_emb, dim_emb),
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index):
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x))
        return out

    def message(self, x_j):
        return torch.nn.functional.relu(x_j)

    def update(self, aggr_out):
        return aggr_out


class GINModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int,
        drop_prob: float = 0.5,
        interleave_diff_graph: bool = False,
        only_original_graph: bool = False,
        only_diff_graph: bool = False,
    ):
        """
        Args
            in_channels: dimension of input features
            hidden_channels: dimension of hidden layers
            num_layers: number of layers
            out_channels: dimension of output
            drop_prob: dropout probability
            interleave_diff_graph: if True, every even layer conv layer message passes on the diffusion graph instead
        """

        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.drop_prob = drop_prob

        assert (
            sum([only_original_graph, only_diff_graph, interleave_diff_graph]) == 1
        ), "Only one of only_original_graph, only_diff_graph, interleave_diff_graph can be used"

        self.interleave_diff_graph = interleave_diff_graph
        self.only_original_graph = only_original_graph
        self.only_diff_graph = only_diff_graph

        self.lin_in = nn.Linear(in_channels, hidden_channels)
        self.pool = global_add_pool
        self.lin_out = nn.Linear(hidden_channels, out_channels)

        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(MyGINConv(hidden_channels))

        self.drop = nn.Dropout(p=drop_prob)

    def forward(self, data):
        x = data.x
        x = self.lin_in(x)

        for i, conv in enumerate(self.convs):
            if self.only_original_graph:
                x = x + conv(x, data.edge_index)
            elif self.only_diff_graph:
                x = x + conv(x, data.rewire_edge_index)
            elif self.interleave_diff_graph:
                if i % 2 == 0:
                    x = x + conv(x, data.edge_index)
                else:
                    x = x + conv(x, data.rewire_edge_index)
            else:
                raise Exception("No message passing scheme chosen")

            x = self.drop(x)

        x = self.pool(x, data.batch)  # (num_nodes, d) -> (batch_size, d)
        x = self.lin_out(x)
        x = x.view(-1)

        return x
