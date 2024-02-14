import torch
from torch_geometric.data import Data
from torch_geometric.datasets import ZINC
from tqdm import tqdm

from data import DoubleExp


def get_data_double_exp(
    rewirer, train_size, val_size, c1, c2, c3, d, seed, device, verbose=False
):
    zinc_graphs_train = ZINC(
        root="data/data_zinc",
        subset=True,
        split="train",
    )

    zinc_graphs_val = ZINC(
        root="data/data_zinc",
        subset=True,
        split="val",
    )

    graphs_train = []
    graphs_val = []

    for i in tqdm(
        range(train_size if train_size != -1 else len(zinc_graphs_train)),
        desc="Generating training data",
        disable=not verbose,
    ):
        g = DoubleExp(
            id=i,
            data=zinc_graphs_train[i],
            c1=c1,
            c2=c2,
            c3=c3,
            d=d,
            seed=seed,
            rewirer=rewirer,
        )
        graphs_train.append(g.to_torch_data().to(device))

    for i in tqdm(
        range(val_size if val_size != -1 else len(zinc_graphs_val)),
        desc="Generating validation data",
        disable=not verbose,
    ):
        g = DoubleExp(
            id=i,
            data=zinc_graphs_val[i],
            c1=c1,
            c2=c2,
            c3=c3,
            d=d,
            seed=seed,
            rewirer=rewirer,
        )
        graphs_val.append(g.to_torch_data().to(device))

    return graphs_train, graphs_val


def main():
    get_data_double_exp()


if __name__ == "__main__":
    main()
