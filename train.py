import torch
import numpy as np

from generate_data import get_data
from models import GINVanilla


# TODO: graph batching
# TODO: train + eval methods with expanders


def train_model(data, model, optimiser, loss_fn):
    model.train()
    optimiser.zero_grad()
    y_pred = model(data.x, data.edge_index)
    loss = loss_fn(y_pred, data.y)
    loss.backward()
    optimiser.step()
    return loss.item()



@torch.no_grad()
def eval_model(data_loader, model, loss_fn):
    model.eval()
    loss = []
    for data in data_loader:
        y_pred = model(data.x, data.edge_index)
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

    dl_train, dl_val = get_data()

    model = GINVanilla(in_channels=1, hidden_channels=16, num_layers=5, out_channels=1)

    train_eval_loop(model, dl_train, dl_val, lr=0.001, num_epochs=10, print_every=1)



if __name__ == "__main__":
    main()
