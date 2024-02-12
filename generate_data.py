# %%

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import ZINC
from tqdm import tqdm

from data import DoubleExp, ExpMix


def make_single_dummy():
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


def make_single_zinc():
    zinc_graphs = ZINC(
        root="data_zinc",
        subset=True,
        split="train",
    )

    g = ExpMix(id=123, data=zinc_graphs[1], seed=123, rewirer="cayley")

    # g.draw_rewire()
    g.draw()
    print(f"{g.num_nodes=} {g.num_edges=} {g.dim_feat=}")
    print(g.x)
    print(g.y)
    print(g.interact_strength)
    print(g.edge_index)
    print(g.rewire_edge_index)

    return g


def get_data(device=None, rewirer=None, size_train=500, size_val=100):
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

    for i in tqdm(range(size_train)):
        g = ExpMix(id=i, data=zinc_graphs_train[i], seed=i, rewirer=rewirer)
        graphs_train.append(g.to_torch_data().to(device))

    for i in tqdm(range(size_val)):
        g = ExpMix(id=i, data=zinc_graphs_val[i], seed=i, rewirer=rewirer)
        graphs_val.append(g.to_torch_data().to(device))

    return graphs_train, graphs_val


def get_data_double_exp(device=None, rewirer=None, size_train=500, size_val=100, c1=0.5, c2=0.5, d=5):
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

    for i in tqdm(range(size_train)):
        g = DoubleExp(
            id=i,
            data=zinc_graphs_train[i],
            c1=c1,
            c2=c2,
            d=d,
            seed=i,
            rewirer=rewirer,
        )
        graphs_train.append(g.to_torch_data().to(device))

    for i in tqdm(range(size_val)):
        g = DoubleExp(
            id=i,
            data=zinc_graphs_val[i],
            c1=c1,
            c2=c2,
            d=d,
            seed=i,
            rewirer=rewirer,
        )
        graphs_val.append(g.to_torch_data().to(device))

    return graphs_train, graphs_val


def main():
    # make_single_dummy()
    # make_single_zinc()
    # get_data()
    # get_data_double_exp()


if __name__ == "__main__":
    main()
# %%
