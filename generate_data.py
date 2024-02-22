import numpy as np
import torch
from torch_geometric.datasets import ZINC
from tqdm import tqdm

from data import ColourInteract, SalientDists


def get_data_SalientDists(
    rewirers: list, train_size, val_size, c1, c2, c3, d, train_test_size_boundary, seed, device, verbose=False
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

    graphs_train = [[] for _ in rewirers]
    graphs_val = [[] for _ in rewirers]

    num_nodes_train = []
    num_nodes_val = []

    for num, rewirer in enumerate(rewirers):

        for i in tqdm(
            range(len(zinc_graphs_train)),
            desc="Generating training data",
            disable=not verbose,
        ):
            if len(graphs_train[num]) < train_size:
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
                if g.num_nodes < train_test_size_boundary:
                    graphs_train[num].append(g.to_torch_data().to(device))
                    num_nodes_train.append(g.num_nodes)

        assert len(graphs_train[num]) == train_size

        print(f"Num nodes for {rewirer}")
        print(
            f"mean: {np.mean(num_nodes_train)}, std: {np.std(num_nodes_train)}, max: {np.max(num_nodes_train)}")

    for num, rewirer in enumerate(rewirers):
        for i in tqdm(
            range(len(zinc_graphs_train)),
            desc="Generating validation data",
            disable=not verbose,
        ):
            if len(graphs_val[num]) < val_size:
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
                if g.num_nodes >= train_test_size_boundary:

                    graphs_val[num].append(g.to_torch_data().to(device))
                    num_nodes_val.append(g.num_nodes)

        print(f"Num nodes for {rewirer}")
        print(
            f"mean: {np.mean(num_nodes_val)}, std: {np.std(num_nodes_val)}, min: {np.min(num_nodes_val)}")

        assert len(
            graphs_val[num]) == val_size, f"len(graphs_val[{num}]) = {len(graphs_val[num])}, val_size = {val_size}"

    return graphs_train, graphs_val


def get_data_ColourInteract(
    rewirers, train_size, val_size, c1, c2, num_colours, seed, device, verbose=False
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

    graphs_train = [[] for _ in rewirers]
    graphs_val = [[] for _ in rewirers]

    num_nodes_train = []

    for i in tqdm(
        range(train_size if train_size != -1 else len(zinc_graphs_train)),
        desc="Generating training data",
        disable=not verbose,
    ):
        for num, rewirer in enumerate(rewirers):

            g = ColourInteract(
                id=i,
                data=zinc_graphs_train[i],
                c1=c1,
                c2=c2,
                num_colours=num_colours,
                seed=seed,
                rewirer=rewirer,
            )

            graphs_train[num].append(g.to_torch_data().to(device))

        num_nodes_train.append(g.num_nodes)

    for i in tqdm(
        range(val_size if val_size != -1 else len(zinc_graphs_val)),
        desc="Generating validation data",
        disable=not verbose,
    ):
        num_nodes_train.append(g.num_nodes)

        g = ColourInteract(
            id=i,
            data=zinc_graphs_val[i],
            c1=c1,
            c2=c2,
            num_colours=num_colours,
            seed=seed,
            rewirer=rewirer,
        )
        graphs_val[num].append(g.to_torch_data().to(device))

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
