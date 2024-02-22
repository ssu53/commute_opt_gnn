import numpy as np
import torch
from torch_geometric.datasets import ZINC
from torch_geometric.datasets import LRGBDataset
from tqdm import tqdm

from data import ColourInteract, SalientDists
from torch_geometric.utils import to_networkx
import networkx as nx


def get_data_SalientDists(
    rewirers: list, dataset: str, train_size, val_size, c1, c2, c3, d, min_train_nodes, max_train_nodes, max_val_nodes, seed, device, verbose=False
):

    print("Loading data...")
    if dataset == "ZINC":
        all_graphs_train = ZINC(
            root="data/data_zinc",
            subset=True,
            split="train",
        )

        all_graphs_val = ZINC(
            root="data/data_zinc",
            subset=True,
            split="val",
        )
    elif dataset == "LRGB":
        all_graphs_train = LRGBDataset(
            root="data/data_lrgb",
            name="Peptides-struct",
            split="train",
        )

        all_graphs_val = LRGBDataset(
            root="data/data_lrgb",
            name="Peptides-struct",
            split="val",
        )

    graphs_train = [[] for _ in rewirers]
    graphs_val = [[] for _ in rewirers]

    num_nodes_train = []
    num_nodes_val = []

    print("Preprocessing data...")

    for num, rewirer in enumerate(rewirers):
        pbar = tqdm(total=train_size,
                    desc=f"Generating training data for {rewirer}", disable=verbose)
        for i in range(len(all_graphs_train)):
            if len(graphs_train[num]) < train_size and nx.is_connected(to_networkx(all_graphs_train[i], to_undirected=True)):
                g = SalientDists(
                    id=i,
                    data=all_graphs_train[i],
                    c1=c1,
                    c2=c2,
                    c3=c3,
                    d=d,
                    seed=seed,
                    rewirer=rewirer,
                )
                if g.num_nodes > min_train_nodes and g.num_nodes < max_train_nodes and nx.diameter(to_networkx(g.data, to_undirected=True)) > d:
                    graphs_train[num].append(g.to_torch_data().to(device))
                    num_nodes_train.append(g.num_nodes)
                    pbar.update(1)

        assert len(graphs_train[num]) == train_size

        print(f"Num nodes for {rewirer}")
        print(
            f"mean: {np.mean(num_nodes_train)}, std: {np.std(num_nodes_train)}, max: {np.max(num_nodes_train)}")

    for num, rewirer in enumerate(rewirers):
        pbar = tqdm(total=val_size,
                    desc=f"Generating validation data for {rewirer}", disable=verbose)
        for i in range(len(all_graphs_val)):
            if len(graphs_val[num]) < val_size and nx.is_connected(to_networkx(all_graphs_val[i], to_undirected=True)):
                g = SalientDists(
                    id=i,
                    data=all_graphs_val[i],
                    c1=c1,
                    c2=c2,
                    c3=c3,
                    d=d,
                    seed=seed,
                    rewirer=rewirer,
                )
                if g.num_nodes < max_val_nodes and nx.diameter(to_networkx(g.data, to_undirected=True)) > d:

                    graphs_val[num].append(g.to_torch_data().to(device))
                    num_nodes_val.append(g.num_nodes)
                    pbar.update(1)

        print(f"Num nodes for {rewirer}")
        print(
            f"mean: {np.mean(num_nodes_val)}, std: {np.std(num_nodes_val)}, min: {np.min(num_nodes_val)}")

        assert len(
            graphs_val[num]) == val_size, f"len(graphs_val[{num}]) = {len(graphs_val[num])}, val_size = {val_size}"

    return graphs_train, graphs_val


def get_data_ColourInteract(
    dataset, rewirers, train_size, val_size, c1, c2, num_colours, min_train_nodes, max_train_nodes, max_val_nodes, seed, device, verbose=False
):
    print("Loading data...")
    if dataset == "ZINC":
        all_graphs_train = ZINC(
            root="data/data_zinc",
            subset=True,
            split="train",
        )

        all_graphs_val = ZINC(
            root="data/data_zinc",
            subset=True,
            split="val",
        )
    elif dataset == "LRGB":
        all_graphs_train = LRGBDataset(
            root="data/data_lrgb",
            name="Peptides-struct",
            split="train",
        )

        all_graphs_val = LRGBDataset(
            root="data/data_lrgb",
            name="Peptides-struct",
            split="val",
        )

    graphs_train = [[] for _ in rewirers]
    graphs_val = [[] for _ in rewirers]

    num_nodes_train = []

    for num, rewirer in enumerate(rewirers):
        pbar = tqdm(total=train_size,
                    desc=f"Generating training data for {rewirer}", disable=verbose)
        for i in range(len(all_graphs_train)):
            if len(graphs_train[num]) < train_size and nx.is_connected(to_networkx(all_graphs_train[i], to_undirected=True)):
                g = ColourInteract(
                    id=i,
                    data=all_graphs_train[i],
                    c1=c1,
                    c2=c2,
                    num_colours=num_colours,
                    seed=seed,
                    rewirer=rewirer,
                )

                if g.num_nodes > min_train_nodes and g.num_nodes < max_train_nodes:
                    graphs_train[num].append(g.to_torch_data().to(device))
                    pbar.update(1)

        num_nodes_train.append(g.num_nodes)
    for num, rewirer in enumerate(rewirers):
        pbar = tqdm(total=train_size,
                    desc=f"Generating validation data for {rewirer}", disable=verbose)
        for i in range(len(all_graphs_val)):
            if len(graphs_val[num]) < val_size and nx.is_connected(to_networkx(all_graphs_val[i], to_undirected=True)):

                g = ColourInteract(
                    id=i,
                    data=all_graphs_val[i],
                    c1=c1,
                    c2=c2,
                    num_colours=num_colours,
                    seed=seed,
                    rewirer=rewirer,
                )
                if g.num_nodes < max_val_nodes:
                    graphs_val[num].append(g.to_torch_data().to(device))
                    pbar.update(1)

    return graphs_train, graphs_val


def main():

    print("Getting small sample of SalientDists...")

    graphs_train, graphs_val = get_data_SalientDists(
        rewirer="fully_connected",
        train_size=100,
        val_size=50,
        c1=0.7,
        c2=0.3,
        c3=0.1,
        d=5,
        seed=42,
        device=torch.device("cpu"),
    )

    print([g.y.cpu() for g in graphs_train])
    print([g.y.cpu() for g in graphs_val])

    print(
        f"train targets: {np.mean([g.y.cpu() for g in graphs_train]):.2f} +/- {np.std([g.y.cpu() for g in graphs_train]):.3f}")
    print(
        f"val targets: {np.mean([g.y.cpu() for g in graphs_val]):.2f} +/- {np.std([g.y.cpu() for g in graphs_val]):.3f}")

    print("Getting small sample of ColourInteract...")

    graphs_train, graphs_val = get_data_ColourInteract(
        rewirer=None,
        train_size=100,
        val_size=50,
        c1=1.0,
        c2=0.5,
        num_colours=8,
        seed=42,
        device=torch.device("cpu"),
    )

    print(
        f"train targets: {np.mean([g.y.cpu() for g in graphs_train]):.2f} +/- {np.std([g.y.cpu() for g in graphs_train]):.3f}")
    print(
        f"val targets: {np.mean([g.y.cpu() for g in graphs_val]):.2f} +/- {np.std([g.y.cpu() for g in graphs_val]):.3f}")


if __name__ == "__main__":
    main()
