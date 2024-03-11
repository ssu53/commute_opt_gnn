# %%

import argparse
import pickle
from pathlib import Path
from pprint import pprint

import torch
import yaml
from easydict import EasyDict
from ogb.nodeproppred import PygNodePropPredDataset
from tqdm import tqdm

import wandb
from gin import GINModel
from train_synthetic import count_parameters, set_seed

import torch.nn.functional as F

def get_ogbn_arxiv():
    """
    see data description at https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv
    """

    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="../data/")

    NUM_CLASSES = 40

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = (
        split_idx["train"],
        split_idx["valid"],
        split_idx["test"],
    )
    graph = dataset[0]  # pyg graph object

    assert len(train_idx) + len(valid_idx) + len(test_idx) == graph.num_nodes
    assert graph.x.size(0) == graph.num_nodes
    assert torch.min(graph.y).item() == 0
    assert torch.max(graph.y).item() == NUM_CLASSES - 1

    print(f"train frac {len(train_idx) / graph.num_nodes : .3f}")
    print(f"valid frac {len(valid_idx) / graph.num_nodes : .3f}")
    print(f"test frac {len(test_idx) / graph.num_nodes : .3f}")

    return graph, train_idx, valid_idx, test_idx, NUM_CLASSES


def get_accuracy(logits, y, batch_size=None):
    """
    specify batch size if logits and y are too large
    """

    assert logits.size(0) == y.size(0)
    num_nodes = logits.size(0)

    if batch_size is None:

        y_hat = torch.argmax(logits, dim=-1, keepdim=True)
        acc = (y == y_hat).sum().item() / num_nodes

    else:

        assert batch_size <= num_nodes

        accs = []
        for batch in range(0, num_nodes, batch_size):
            logits_chunk = logits[batch : batch + batch_size]
            y_chunk = y[batch : batch + batch_size]
            y_hat_chunk = torch.argmax(logits_chunk, dim=-1, keepdim=True)
            accs.append((y_chunk == y_hat_chunk).sum().item())

        print("num batches", len(accs))
        acc = sum(accs) / num_nodes

    return acc


def train_model(
    data, model, mask, optimiser, loss_fn=torch.nn.functional.cross_entropy
):
    model.train()
    optimiser.zero_grad()
    y = data.y[mask]
    optimiser.zero_grad()
    logits = model(data)[mask]
    if y.ndim > 1:
        assert (y.ndim == 2) and (y.size(1) == 1)
        y = y.squeeze(dim=-1)
    loss = loss_fn(logits, y)
    loss.backward()
    optimiser.step()
    del logits
    torch.cuda.empty_cache()
    return loss.item()


@torch.no_grad()
def eval_model(data, model, mask):
    if sum(mask).item() == 0:
        return torch.nan
    model.eval()
    y = data.y[mask]
    logits = model(data)[mask]
    acc = get_accuracy(logits, y)
    del logits
    torch.cuda.empty_cache()
    return acc


def train_eval_loop(
    model,
    data,
    train_idx,
    val_idx,
    # test_mask,
    lr: float,
    num_epochs: int,
    print_every: int,
    verbose: bool = False,
    log_wandb: bool = True,
):

    # optimiser and loss function
    # -------------------------------

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(
        reduction="mean"
    )  # xent loss for multiclass classification

    # dicts to store training stats
    # -------------------------------

    epoch2trainloss = {}
    epoch2trainacc = {}
    epoch2valacc = {}

    # initial metrics
    # -------------------------------

    train_acc = eval_model(data, model, train_idx)
    val_acc = eval_model(data, model, val_idx)
    # test_acc = eval_model(data, model, test_mask)

    epoch = 0
    epoch2trainacc[epoch] = train_acc
    epoch2valacc[epoch] = val_acc

    if log_wandb:
        wandb.log(
            {
                "train/acc": train_acc,
                "val/acc": val_acc,
                # "test/acc": test_acc,
            }
        )

    # training loop
    # -------------------------------

    print("Whole batch gradient descent on one graph...")

    with tqdm(
        range(1, num_epochs + 1), unit="e", disable=not verbose
    ) as tepoch:
        # with tqdm(range(1,num_epochs+1), unit="e", disable=not verbose) as tepoch:
        for epoch in tepoch:

            train_loss = train_model(data, model, train_idx, optimiser, loss_fn)

            if log_wandb and not epoch % print_every == 0:
                wandb.log({"train/loss": train_loss})

            if epoch % print_every == 0:

                train_acc = eval_model(data, model, train_idx)
                val_acc = eval_model(data, model, val_idx)

                epoch2trainloss[epoch] = train_loss
                epoch2trainacc[epoch] = train_acc
                epoch2valacc[epoch] = val_acc

                if log_wandb:
                    wandb.log(
                        {
                            "train/loss": train_loss,
                            "train/acc": train_acc,
                            "val/acc": val_acc,
                        }
                    )

                tepoch.set_postfix(
                    train_loss=train_loss, train_acc=train_acc, val_acc=val_acc
                )

    # final metrics
    # -------------------------------

    train_acc = eval_model(data, model, train_idx)
    val_acc = eval_model(data, model, val_idx)
    # test_acc = eval_model(data, model, test_mask)

    epoch = num_epochs
    epoch2trainacc[epoch] = train_acc
    epoch2valacc[epoch] = val_acc

    if log_wandb:
        wandb.log(
            {
                "train/acc": train_acc,
                "val/acc": val_acc,
                # "test/acc": test_acc,
            }
        )

    # print metrics
    # -------------------------------

    print(
        f"Max val acc at epoch {max(epoch2valacc, key=epoch2valacc.get)}: {max(epoch2valacc.values())}"
    )
    print(f"Final val acc at epoch {num_epochs}: {epoch2valacc[num_epochs]}")

    if verbose:
        print("epoch: train loss | train acc | val acc")
        print(
            "\n".join(
                "{!r}: {:.5f} | {:.3f} | {:.3f}".format(
                    epoch,
                    epoch2trainloss[epoch],
                    epoch2trainacc[epoch],
                    epoch2valacc[epoch],
                )
                for epoch in epoch2trainloss  # this doesn't print the 0th epoch before training
            )
        )

    end_results = {
        "end": val_acc,
        "best": max(epoch2valacc.values()),
    }

    del optimiser
    del loss_fn

    return end_results


def get_rewire_edge_index(rewirer: str):
    """
    these are various ways to instantiate Cayley clusters on this dataset
    """

    base_rewire_dir = Path("../data/arxiv-rewirings")

    if rewirer == "class_all":
        fn = "arxiv_rewire_by_class_all"
    elif rewirer == "class_train_only":
        fn = "arxiv_rewire_by_class_train_only"
    elif rewirer == "kmeans_all":
        fn = "arxiv_rewire_by_kmeans_all"
    elif rewirer == "mlp_all":
        fn = "arxiv_rewire_by_mlp_all"
    elif rewirer == "mlp_feats_all":
        fn = "arxiv_rewire_by_mlp_feats_all"
    elif rewirer == "enriched-kmeans_all":
        fn = "arxiv_rewire_by_enriched-kmeans_all"
    elif rewirer == "corrupted_class_all":
        fn = "arxiv_rewire_by_corrupted_class_all"
    elif rewirer == "corrupted_0_5_class_all":
        fn = "arxiv_rewire_by_corrupted_0_5_class_all"
    else:
        raise NotImplementedError

    print(f"Using rewiring from: {fn}")

    with open(base_rewire_dir / fn, "rb") as f:
        rewire_edge_index = pickle.load(f)

    return rewire_edge_index


def main(config):

    pprint(config)

    # set device
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # load corrupted dataset
    # -------------------------------
    # graph, train_idx, valid_idx, test_idx, num_classes = get_ogbn_arxiv()

    with open("../data/ogbn_arxiv/corrupted_0_5_data.pkl", "rb") as f:
        save_dict = pickle.load(f)

    graph, train_idx, valid_idx, test_idx, num_classes, corrupted_y = (
        save_dict["graph"],
        save_dict["train_idx"],
        save_dict["valid_idx"],
        save_dict["test_idx"],
        save_dict["num_classes"],
        save_dict["corrupted_y"],
    )

    config.model.in_channels = graph.x.size(1)
    config.model.out_channels = num_classes


    if config.model.rewirer is None:
        print("Adding one-hot corrupted labels to features")
        one_hot_corrupted = F.one_hot(corrupted_y.squeeze(), num_classes=40).float()
        graph.x = torch.cat([graph.x, one_hot_corrupted], dim=1)
        print("New feature shape", graph.x.shape)
    
    # attach the rewirer
    # -------------------------------
    if config.model.rewirer is not None:
        graph.rewire_edge_index = get_rewire_edge_index(config.model.rewirer)

    # get moodel
    # -------------------------------

    set_seed(config.model.seed)

    model = GINModel(
        in_channels=graph.x.shape[1],
        hidden_channels=config.model.hidden_channels,
        num_layers=config.model.num_layers,
        out_channels=config.model.out_channels,
        drop_prob=config.model.drop_prob,
        only_original_graph=(config.model.approach == "only_original"),
        interleave_diff_graph=(config.model.approach == "interleave"),
        only_diff_graph=(config.model.approach == "only_diff_graph"),
        global_pool_aggr=None,
        norm=config.model.norm,
    )

    count_parameters(model)
    print(model)

    # train
    # -------------------------------

    model.to(device)
    graph.to(device)

    if config.train.log_wandb:
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            config=config,
            group=f"rewirer-{config.model.rewirer}-{config.model.approach}",
        )
        wandb.run.name = f"corrupt-prob-0.5-rewirer-{config.model.rewirer}-{config.model.approach}-seed-{config.model.seed}"

    end_results = train_eval_loop(
        model,
        graph,
        train_idx,
        valid_idx,
        lr=config.train.lr,
        num_epochs=config.train.num_epochs,
        print_every=config.train.print_every,
        verbose=config.train.verbose,
        log_wandb=config.train.log_wandb,
    )

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_fn",
        default="../configs/debug_ogbn-arxiv.yaml",
        help="configuration file name",
        type=str,
    )

    args = parser.parse_args()

    with open(args.config_fn, "r") as f:
        config = EasyDict(yaml.safe_load(f))

    for approach in config.model.approaches:
        if approach == "only_original":
            rewirers = [None]
        else:
            rewirers = config.model.rewirers
        for rewirer in rewirers:
            config.model.rewirer = rewirer
            config.model.approach = approach
            main(config)
    

# def exploratory_stuff_to_clean_up():

#     graph, train_idx, valid_idx, test_idx, num_classes = get_ogbn_arxiv()

#     import matplotlib.pyplot as plt
#     from sklearn.cluster import KMeans
#     from sklearn.decomposition import PCA

#     features = graph.x[train_idx]

#     num_clusters = 10
#     kmeans_model = KMeans(num_clusters, random_state=0, n_init="auto").fit(features)
#     cluster_labels = kmeans_model.predict(features)

#     pca_model = PCA(n_components=2)  # visualise in 2D
#     features_pca = pca_model.fit_transform(features)

#     plt.figure()
#     for i in range(num_clusters):
#         plt.scatter(
#             features_pca[cluster_labels == i, 0],
#             features_pca[cluster_labels == i, 1],
#             label=f"Cluster {i}",
#             alpha=0.5,
#         )
#     plt.show()

#     from collections import Counter

#     import pandas as pd
#     import seaborn as sns

#     ys = graph.y[train_idx]

#     df = pd.DataFrame(index=range(num_classes), columns=range(num_clusters), dtype=int)
#     df.index.name = "class"

#     for i in range(num_clusters):
#         cnt = Counter(ys[cluster_labels == i].flatten().tolist())
#         cnt.subtract({label: 0 for label in range(num_classes)})
#         df[i] = cnt  # for zero counts

#     plt.figure(figsize=(5, 10))
#     sns.heatmap(df, annot=True, cmap="viridis", fmt="", cbar=False)
#     plt.show()

#     df_norm_cluster = df / df.sum()

#     plt.figure(figsize=(5, 10))
#     sns.heatmap(df_norm_cluster, annot=True, cmap="viridis", fmt=".2f", cbar=False)
#     plt.show()

#     df_norm_class = df.divide(df.sum(axis=1), axis=0)

#     plt.figure(figsize=(5, 10))
#     sns.heatmap(df_norm_class, annot=True, cmap="viridis", fmt=".2f", cbar=False)
#     plt.show()

#     # ----------

#     from compute_arxiv_rewire import check_valid

#     graph, train_idx, valid_idx, test_idx, num_classes = get_ogbn_arxiv()

#     # the allowable indices are any!
#     # here we allow it to look at val and test to draw expander
#     rewire_edge_index = get_rewire_edge_index(rewirer="by_class_all")
#     check_valid(
#         rewire_edge_index, graph.y.squeeze(), torch.tensor(range(graph.num_nodes))
#     )

#     # the allowable indices are in train_idx only!
#     rewire_edge_index = get_rewire_edge_index(rewirer="by_class_train_only")
#     check_valid(rewire_edge_index, graph.y.squeeze(), train_idx)

# %%
