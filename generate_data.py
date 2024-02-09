import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data import DoubleExp, ExpMix


def make_dummy():
    dummy_data = Data(
        x=torch.zeros((5, 1)),
        y=None,
        edge_attr=None,
        edge_index=torch.tensor([[0, 0, 0, 2], [1, 2, 3, 4]]),
    )

    g = ExpMix(id="dummy", data=dummy_data, seed=0)

    g.draw()
    print(f"{g.num_nodes=} {g.num_edges=} {g.dim_feat=}")
    print(g.x)
    print(g.y)
    print(g.interact_strength)

    print("returning g")
    return g


def make_zinc():
    zinc_graphs = ZINC(
        root="data_zinc",
        subset=True,
        split="train",
    )

    g = ExpMix(id=123, data=zinc_graphs[1], seed=123, expander="cayley")

    # g.draw_expander()
    g.draw()
    print(f"{g.num_nodes=} {g.num_edges=} {g.dim_feat=}")
    print(g.x)
    print(g.y)
    print(g.interact_strength)
    print(g.edge_index)
    print(g.expander_edge_index)


def get_data(device=None, expander=None):
    if device is None:
        device = torch.device("cpu")

    zinc_graphs = ZINC(
        root="data_zinc",
        subset=True,
        split="train",
    )

    graphs_train = []
    graphs_val = []

    for i in range(500):
        g = ExpMix(id=i, data=zinc_graphs[i], seed=i, expander=expander)
        graphs_train.append(g.to_torch_data().to(device))

    for i in range(500, 600):
        g = ExpMix(id=i, data=zinc_graphs[i], seed=i, expander=expander)
        graphs_val.append(g.to_torch_data().to(device))

    # import numpy as np
    # print(np.mean([g.y for g in graphs_train]), np.std([g.y for g in graphs_train]))
    # print(np.mean([g.y for g in graphs_val]), np.std([g.y for g in graphs_val]))

    # can't batch differently sized graphs, need to implement
    dl_train = DataLoader(graphs_train, batch_size=1)
    dl_val = DataLoader(graphs_val, batch_size=1)

    return dl_train, dl_val


def get_data_double_exp(device=None, expander=None):
    if device is None:
        device = torch.device("cpu")

    zinc_graphs_train = ZINC(
        root="data_zinc",
        subset=True,
        split="train",
    )

    zinc_graphs_val = ZINC(
        root="data_zinc",
        subset=True,
        split="val",
    )

    graphs_train = []
    graphs_val = []

    c1, c2, d = 0.5, 0.5, 5

    for i in tqdm(range(500)):
        g = DoubleExp(
            id=i,
            data=zinc_graphs_train[i],
            c1=c1,
            c2=c2,
            d=d,
            seed=i,
            expander=expander,
        )
        graphs_train.append(g.to_torch_data().to(device))

    for i in tqdm(range(100)):
        g = DoubleExp(
            id=i,
            data=zinc_graphs_val[i],
            c1=c1,
            c2=c2,
            d=d,
            seed=i,
            expander=expander,
        )
        graphs_val.append(g.to_torch_data().to(device))

    print(np.mean([g.y for g in graphs_train]), np.std([g.y for g in graphs_train]))
    print(np.mean([g.y for g in graphs_val]), np.std([g.y for g in graphs_val]))

    # can't batch differently sized graphs, need to implement
    dl_train = DataLoader(graphs_train, batch_size=1)
    dl_val = DataLoader(graphs_val, batch_size=1)

    return dl_train, dl_val


def main():
    # make_dummy()
    # make_zinc()
    # get_data()
    get_data_double_exp()


if __name__ == "__main__":
    main()
# %%
