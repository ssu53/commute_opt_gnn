import torch
import numpy as np

from generate_data import get_data_double_exp
from torch_geometric.loader import DataLoader
from models import GINModel




def train_model(data, model, optimiser, loss_fn):
    model.train()
    optimiser.zero_grad()
    y_pred = model(data)
    loss = loss_fn(y_pred, data.y)
    loss.backward()
    optimiser.step()
    return loss.item()



@torch.no_grad()
def eval_model(data_loader, model, loss_fn):
    model.eval()
    loss = []
    for data in data_loader:
        y_pred = model(data)
        loss.append(loss_fn(y_pred, data.y).item())
    return np.mean(loss)



def train_eval_loop(model, data_loader_train, data_loader_val, lr: float, num_epochs: int, loss_fn=None, print_every=10):
    
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    
    if loss_fn is None:
        loss_fn = torch.nn.MSELoss(reduction='mean')
        print("Using MSE Loss...")
    
    for epoch in range(1,num_epochs+1):
        for data_train in data_loader_train:
            train_loss = train_model(data_train, model, optimiser, loss_fn)
        if epoch % print_every == 0:
            val_loss = eval_model(data_loader_val, model, loss_fn)
            print(f"Epoch {epoch} | train loss {train_loss:.3f} | val loss {val_loss:.3f}")

    return



def main():

    # TODO: configure args
    rewirer = "cayley"
    c1 = 0.7
    c2 = 0.3
    d = 5
    batch_size = 32
    hidden_channels = 8
    num_layers = 5
    drop_prob = 0.0
    lr = 0.001
    num_epochs = 100
    print_every = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(
        "CONFIGS\n", 
        f"{rewirer=}\n",
        f"{c1=}\n",
        f"{c2=}\n",
        f"{d=}\n",
        f"{batch_size=}\n",
        f"{hidden_channels=}\n",
        f"{num_layers=}\n",
        f"{drop_prob=}\n",
        f"{lr=}\n",
        f"{num_epochs=}\n",
        f"{print_every=}\n",
    )

    
    # Get data

    graphs_train, graphs_val = get_data_double_exp(rewirer=rewirer, device=device, c1=c1, c2=c2, d=d)

    print(f"train targets: {np.mean([g.y.cpu() for g in graphs_train]):.2f} +/- {np.std([g.y.cpu() for g in graphs_train]):.3f}")
    print(f"val targets: {np.mean([g.y.cpu() for g in graphs_val]):.2f} +/- {np.std([g.y.cpu() for g in graphs_val]):.3f}")

    dl_train = DataLoader(graphs_train, batch_size=batch_size)
    dl_val = DataLoader(graphs_val, batch_size=batch_size)

    in_channels = graphs_train[0].x.shape[1]
    out_channels = 1 if len(graphs_train[0].y.shape) == 0 else len(graphs_train[0].y.shape)


    # Train

    print("Training a GIN model without rewiring...")

    model = GINModel(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers, out_channels=out_channels, drop_prob=drop_prob, interleave_diff_graph=False)
    model.to(device)

    train_eval_loop(model, dl_train, dl_val, lr=lr, num_epochs=num_epochs, print_every=print_every)

    print("Training a GIN model with interleaved rewiring...")

    model = GINModel(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers, out_channels=out_channels, drop_prob=drop_prob, interleave_diff_graph=True)
    model.to(device)

    train_eval_loop(model, dl_train, dl_val, lr=lr, num_epochs=num_epochs, print_every=print_every)



if __name__ == "__main__":
    main()
