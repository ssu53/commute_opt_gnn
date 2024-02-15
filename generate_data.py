import torch
from torch_geometric.datasets import ZINC
from tqdm import tqdm

from data import SalientDists, ColourInteract

import numpy as np


def get_data_SalientDists(
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

    num_nodes_train = []

    for i in tqdm(
        range(train_size if train_size != -1 else len(zinc_graphs_train)),
        desc="Generating training data",
        disable=not verbose,
    ):
        g = SalientDists(
            id=i,
            data=zinc_graphs_train[i],
            c1=c1,
            c2=c2,
            c3=c3,
            d=d,
            seed=seed,
            rewirer=rewirer,
        )

        num_nodes_train.append(g.num_nodes)

        graphs_train.append(g.to_torch_data().to(device))

    print("NUMBER OF NODES")

    print(f"mean: {np.mean(num_nodes_train)}, std: {np.std(num_nodes_train)}")

    for i in tqdm(
        range(val_size if val_size != -1 else len(zinc_graphs_val)),
        desc="Generating validation data",
        disable=not verbose,
    ):
        g = SalientDists(
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



def get_data_ColourInteract(
    rewirer, train_size, val_size, c1, c2, num_colours, seed, device, verbose=False
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

    num_nodes_train = []

    for i in tqdm(
        range(train_size if train_size != -1 else len(zinc_graphs_train)),
        desc="Generating training data",
        disable=not verbose,
    ):
        g = ColourInteract(
            id=i,
            data=zinc_graphs_train[i],
            c1=c1,
            c2=c2,
            num_colours=num_colours,
            seed=seed,
            rewirer=rewirer,
        )

        num_nodes_train.append(g.num_nodes)

        graphs_train.append(g.to_torch_data().to(device))
    

    for i in tqdm(
        range(val_size if val_size != -1 else len(zinc_graphs_val)),
        desc="Generating validation data",
        disable=not verbose,
    ):
        g = ColourInteract(
            id=i,
            data=zinc_graphs_val[i],
            c1=c1,
            c2=c2,
            num_colours=num_colours,
            seed=seed,
            rewirer=rewirer,
        )
        graphs_val.append(g.to_torch_data().to(device))

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

    print(f"train targets: {np.mean([g.y.cpu() for g in graphs_train]):.2f} +/- {np.std([g.y.cpu() for g in graphs_train]):.3f}")
    print(f"val targets: {np.mean([g.y.cpu() for g in graphs_val]):.2f} +/- {np.std([g.y.cpu() for g in graphs_val]):.3f}")



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


    print(f"train targets: {np.mean([g.y.cpu() for g in graphs_train]):.2f} +/- {np.std([g.y.cpu() for g in graphs_train]):.3f}")
    print(f"val targets: {np.mean([g.y.cpu() for g in graphs_val]):.2f} +/- {np.std([g.y.cpu() for g in graphs_val]):.3f}")


if __name__ == "__main__":
    main()
