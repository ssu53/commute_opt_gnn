import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.nn.conv.gin_conv import GINConv



class GINVanilla(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, out_channels: int, eps: float = 0.0, bias: bool = True):
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
            self.lin_layers += [GINConv(nn.Linear(in_channels, hidden_channels, bias), eps)]
            self.lin_layers += [GINConv(nn.Linear(hidden_channels, hidden_channels, bias), eps) for _ in range(num_layers-2)]
            self.lin_layers += [GINConv(nn.Linear(hidden_channels, out_channels, bias), eps)]
        else:
            self.lin_layers += [GINConv(nn.Linear(in_channels, out_channels, bias), eps)]
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
        
        x = self.pool(x, data.batch) # (num_nodes, d) -> (batch_size, d)
        x = x.view(-1) # (batch_size, d) -> (batch_size,)
        
        return x



class GINExpander(nn.Module):
    
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, out_channels: int, eps: float = 0.0, bias: bool = True):
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
            self.lin_layers += [GINConv(nn.Linear(in_channels, hidden_channels, bias), eps)]
            self.lin_layers += [GINConv(nn.Linear(hidden_channels, hidden_channels, bias), eps) for _ in range(num_layers-2)]
            self.lin_layers += [GINConv(nn.Linear(hidden_channels, out_channels, bias), eps)]
        else:
            self.lin_layers += [GINConv(nn.Linear(in_channels, out_channels, bias), eps)]
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
        
        x = self.pool(x, data.batch) # (num_nodes, d) -> (batch_size, d)
        x = x.view(-1) # (batch_size, d) -> (batch_size,)
        
        return x



class MyGINConv(MessagePassing):
    def __init__(self, dim_emb):
        """
        GINConv with built-in MLP (linear -> batch norm -> relu -> linear)
        Args
            dim_emb: embedding dimension
        """

        super().__init__(aggr="add")

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_emb, 2*dim_emb),
            torch.nn.BatchNorm1d(2*dim_emb),
            torch.nn.ReLU(),
            torch.nn.Linear(2*dim_emb, dim_emb))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index):
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x))
        return out

    def message(self, x_j):
        return torch.nn.functional.relu(x_j)

    def update(self, aggr_out):
        return aggr_out



class GINModel(nn.Module):
    
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, out_channels: int, drop_prob: float = 0.5, interleave_diff_graph: bool = False):
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
        self.interleave_diff_graph = interleave_diff_graph
        
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

        for i,conv in enumerate(self.convs):
            if i % 2 == 0:
                x = x + conv(x, data.edge_index)
            else:
                x = x + conv(x, data.expander_edge_index if self.interleave_diff_graph else data.edge_index)
            x = self.drop(x)
        
        x = self.pool(x, data.batch) # (num_nodes, d) -> (batch_size, d)
        x = self.lin_out(x)
        x = x.view(-1)

        return x
