import random
from collections import Counter

import networkx as nx
import numpy as np
import torch
from torch_geometric.datasets import ZINC, LRGBDataset
from torch_geometric.utils import to_networkx
from tqdm import tqdm

from data import ColourInteract, SalientDists


def get_data_SalientDists(
    dataset: str,
    train_size,
    val_size,
    c1,
    c2,
    c3,
    d,
    normalise,
    min_train_nodes,
    max_train_nodes,
    min_val_nodes,
    max_val_nodes,
    verbose=False,
):
    print("Loading data...")
    if dataset == "ZINC":
        all_graphs_train = ZINC(
            root="data/data_zinc",
            subset=False,
            split="train",
        )

        all_graphs_val = ZINC(
            root="data/data_zinc",
            subset=False,
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
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented")


    graphs_train = []
    graphs_val = []

    num_bins = (max_train_nodes-min_train_nodes)//5

    graphs_train_bins = {i: [] for i in range(min_train_nodes//5, (min_train_nodes//5)+num_bins)}
    print(graphs_train_bins)

    for graph in all_graphs_train:
        if graph.num_nodes // 5 in graphs_train_bins:
            nx_graph = to_networkx(graph, to_undirected=True)
            if nx.is_connected(nx_graph):
                graphs_train_bins[graph.num_nodes // 5].append(graph)

    all_graphs_train = []
    for _, graphs in graphs_train_bins.items():
        all_graphs_train.extend(random.sample(graphs, train_size//num_bins))

    print(f"Number of training graphs: {len(all_graphs_train)}")
    print(f"Train graph sizes: {Counter([graph.num_nodes for graph in all_graphs_train])}")
    assert len(all_graphs_train) == train_size, f"len(all_graphs_train) = {len(all_graphs_train)}, train_size = {train_size}"


    pbar = tqdm(
        total=train_size,
        desc=f"Generating training data",
        disable=verbose,
    )
    for i in range(len(all_graphs_train)):
        g = SalientDists(
            id=i,
            data=all_graphs_train[i],
            c1=c1,
            c2=c2,
            c3=c3,
            d=d,
            normalise=normalise,
        )

        graphs_train.append(g)
        pbar.update(1)

    assert (
        len(graphs_train) == train_size
    ), f"len(graphs_train) = {len(graphs_train)}, train_size = {train_size}"



    num_bins = (max_val_nodes - min_val_nodes)//5
    graphs_val_bins = {i: [] for i in range(min_val_nodes//5, (min_val_nodes//5)+num_bins)}
    print(graphs_val_bins)

    for graph in all_graphs_val:
        if graph.num_nodes // 5 in graphs_val_bins:
            nx_graph = to_networkx(graph, to_undirected=True)
            if nx.is_connected(nx_graph):
                graphs_val_bins[graph.num_nodes // 5].append(graph)

    all_graphs_val = []
    for _, graphs in graphs_val_bins.items():
        try:
            all_graphs_val.extend(random.sample(graphs, val_size//num_bins))
        except:
            raise Exception(f"len(graphs) of size {_} = {len(graphs)}, val_size//num_bins = {val_size//num_bins}")


    print(f"Number of validation graphs: {len(all_graphs_train)}")
    print(f"Val graph sizes: {Counter([graph.num_nodes for graph in all_graphs_val])}")
    assert len(all_graphs_val) == val_size, f"len(all_graphs_val) = {len(all_graphs_val)}, val_size = {val_size}"


    pbar = tqdm(
        total=val_size,
        desc=f"Generating validation data",
        disable=verbose,
    )
    for i in range(len(all_graphs_val)):
        g = SalientDists(
            id=i,
            data=all_graphs_val[i],
            c1=c1,
            c2=c2,
            c3=c3,
            d=d,
            normalise=normalise,
        )

        graphs_val.append(g)
        pbar.update(1)

    assert (
        len(graphs_val) == val_size
    ), f"len(graphs_val) = {len(graphs_val)}, val_size = {val_size}"


    return graphs_train, graphs_val


def get_data_ColourInteract(
    dataset,
    train_size,
    val_size,
    c1,
    c2,
    normalise,
    num_colours,
    min_train_nodes,
    max_train_nodes,
    min_val_nodes,
    max_val_nodes,
    verbose=False,
):
    print("Loading data...")
    if dataset == "ZINC":
        all_graphs_train = ZINC(
            root="data/data_zinc",
            subset=False,
            split="train",
        )

        all_graphs_val = ZINC(
            root="data/data_zinc",
            subset=False,
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

    graphs_train = []
    graphs_val = []

    num_bins = (max_train_nodes-min_train_nodes)//5

    graphs_train_bins = {i: [] for i in range(min_train_nodes//5, (min_train_nodes//5)+num_bins)}

    for graph in all_graphs_train:
        if graph.num_nodes // 5 in graphs_train_bins:
            nx_graph = to_networkx(graph, to_undirected=True)
            if nx.is_connected(nx_graph):
                graphs_train_bins[graph.num_nodes // 5].append(graph)

    all_graphs_train = []
    for _, graphs in graphs_train_bins.items():
        all_graphs_train.extend(random.sample(graphs, train_size//num_bins))

    print(f"Number of training graphs: {len(all_graphs_train)}")
    print(f"Train graph sizes: {Counter([graph.num_nodes for graph in all_graphs_train])}")
    assert len(all_graphs_train) == train_size, f"len(all_graphs_train) = {len(all_graphs_train)}, train_size = {train_size}"


    pbar = tqdm(
        total=train_size,
        desc=f"Generating training data",
        disable=verbose,
    )
    for i in range(len(all_graphs_train)):
        g = ColourInteract(
            id=i,
            data=all_graphs_train[i],
            c1=c1,
            c2=c2,
            num_colours=num_colours,
            normalise=normalise,
        )

        graphs_train.append(g)
        pbar.update(1)

    assert (
        len(graphs_train) == train_size
    ), f"len(graphs_train) = {len(graphs_train)}, train_size = {train_size}"


    num_bins = (max_val_nodes - min_val_nodes)//5
    graphs_val_bins = {i: [] for i in range(min_val_nodes//5, (min_val_nodes//5)+num_bins)}
    print(graphs_val_bins)

    for graph in all_graphs_val:
        if graph.num_nodes // 5 in graphs_val_bins:
            nx_graph = to_networkx(graph, to_undirected=True)
            if nx.is_connected(nx_graph):
                graphs_val_bins[graph.num_nodes // 5].append(graph)

    all_graphs_val = []
    for _, graphs in graphs_val_bins.items():
        try:
            all_graphs_val.extend(random.sample(graphs, val_size//num_bins))
        except:
            raise Exception(f"len(graphs) of size {_} = {len(graphs)}, val_size//num_bins = {val_size//num_bins}")


    print(f"Number of validation graphs: {len(all_graphs_train)}")
    print(f"Val graph sizes: {Counter([graph.num_nodes for graph in all_graphs_val])}")
    assert len(all_graphs_val) == val_size, f"len(all_graphs_val) = {len(all_graphs_val)}, val_size = {val_size}"


    pbar = tqdm(
        total=val_size,
        desc=f"Generating validation data",
        disable=verbose,
    )
    for i in range(len(all_graphs_val)):
        g = ColourInteract(
            id=i,
            data=all_graphs_val[i],
            c1=c1,
            c2=c2,
            num_colours=num_colours,
            normalise=normalise
        )

        graphs_val.append(g)
        pbar.update(1)

    assert (
        len(graphs_val) == val_size
    ), f"len(graphs_val) = {len(graphs_val)}, val_size = {val_size}"

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
        f"train targets: {np.mean([g.y.cpu() for g in graphs_train]):.2f} +/- {np.std([g.y.cpu() for g in graphs_train]):.3f}"
    )

    print(
        f"val targets: {np.mean([g.y.cpu() for g in graphs_val]):.2f} +/- {np.std([g.y.cpu() for g in graphs_val]):.3f}"
    )

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
        f"train targets: {np.mean([g.y.cpu() for g in graphs_train]):.2f} +/- {np.std([g.y.cpu() for g in graphs_train]):.3f}"
    )
    print(
        f"val targets: {np.mean([g.y.cpu() for g in graphs_val]):.2f} +/- {np.std([g.y.cpu() for g in graphs_val]):.3f}"
    )


if __name__ == "__main__":
    main()
