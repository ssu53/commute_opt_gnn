import torch
import numpy as np

from generate_data import get_data
from models import GINVanilla, GINExpander


# TODO: graph batching


def train_model(data, model, optimiser, loss_fn):
    model.train()
    optimiser.zero_grad()
    if isinstance(model, GINVanilla):
        y_pred = model(data.x, data.edge_index)
    elif isinstance(model, GINExpander):
        y_pred = model(data.x, data.edge_index, data.expander_edge_index)
    else:
        raise NotImplementedError
    loss = loss_fn(y_pred, data.y)
    loss.backward()
    optimiser.step()
    return loss.item()



@torch.no_grad()
def eval_model(data_loader, model, loss_fn):
    model.eval()
    loss = []
    for data in data_loader:
        if isinstance(model, GINVanilla):
            y_pred = model(data.x, data.edge_index)
        elif isinstance(model, GINExpander):
            y_pred = model(data.x, data.edge_index, data.expander_edge_index)
        else:
            raise NotImplementedError
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

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dl_train, dl_val = get_data(expander="topk")

    for data in dl_train:
        print(data.expander_edge_index)
        break

    print("Training a GIN model without expanders...")

    model = GINVanilla(in_channels=16, hidden_channels=8, num_layers=5, out_channels=1)

    train_eval_loop(model, dl_train, dl_val, lr=0.001, num_epochs=100, print_every=10)

    print("Training a GIN model with expanders...")

    model = GINExpander(in_channels=16, hidden_channels=8, num_layers=5, out_channels=1)

    train_eval_loop(model, dl_train, dl_val, lr=0.001, num_epochs=100, print_every=10)



if __name__ == "__main__":
    main()
