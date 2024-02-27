import argparse
import json
import os
from pprint import pprint

import numpy as np
import torch
import yaml
from easydict import EasyDict
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import wandb
from generate_data import get_data_ColourInteract, get_data_SalientDists
from models import GINModel


def train_model(data, model, optimiser, loss_fn):
    model.train()
    optimiser.zero_grad()
    y_pred = model(data)
    loss = loss_fn(y_pred, data.y)
    loss.backward()
    optimiser.step()
    return loss.item()


@torch.no_grad()
def eval_model(data_loader, model, loss_fn):
    model.eval()
    loss = []
    for data in data_loader:
        y_pred = model(data)
        loss.append(loss_fn(y_pred, data.y).item())
    return np.mean(loss)


def train_eval_loop(
    model,
    data_loader_train,
    data_loader_val,
    lr: float,
    num_epochs: int,
    print_every,
    loss_fn=None,
    verbose=False,
    log_wandb=True,
):
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    if loss_fn is None:
        loss_fn = torch.nn.MSELoss(reduction="mean")
        if verbose:
            print("Using MSE Loss...")

    epoch2trainloss = {}
    epoch2valloss = {}

    with tqdm(range(1, num_epochs + 1), unit="e", disable=not verbose) as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            for data_train in data_loader_train:
                train_loss = train_model(data_train, model, optimiser, loss_fn)
            if epoch % print_every == 0:
                val_loss = eval_model(data_loader_val, model, loss_fn)
                tepoch.set_postfix(train_loss=train_loss, val_loss=val_loss)
                epoch2trainloss[epoch] = train_loss
                epoch2valloss[epoch] = val_loss
                if log_wandb:
                    wandb.log({"train/loss": train_loss, "eval/loss": val_loss})
            else:
                if log_wandb:
                    wandb.log({"train/loss": train_loss})

    print(
        f"Minimum validation loss at epoch {min(epoch2valloss, key=epoch2valloss.get)}: {min(epoch2valloss.values())}"
    )

    print(f"Final loss at epoch {num_epochs}: {min(epoch2valloss.values())}")

    if verbose:
        print("Train / Validation loss by epoch")
        print(
            "\n".join(
                "{!r}: {:.3f} / {:.3f}".format(
                    epoch, epoch2trainloss[epoch], epoch2valloss[epoch]
                )
                for epoch in epoch2trainloss
            )
        )

    end_results = {"end": val_loss, "best": min(epoch2valloss.values())}
    return end_results


def quick_run(rewirers, config_file="debug_ColourInteract.yaml"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42

    with open(f"configs/{config_file}", "r") as f:
        config = EasyDict(yaml.safe_load(f))

    print(
        "CONFIGS\n",
        f"{config_file=}\n",
        f"{rewirers=}\n",
        f"{device=}\n",
        f"{seed=}\n",
        config,
    )

    # Get data

    if config_file == "debug_SalientDists.yaml":
        graphs_train, graphs_val = get_data_SalientDists(
            rewirers=rewirers,
            dataset=config.data.dataset,
            device=device,
            c1=config.data.c1,
            c2=config.data.c2,
            c3=config.data.c3,
            d=config.data.d,
            min_train_nodes=config.data.min_train_nodes,
            max_train_nodes=config.data.max_train_nodes,
            max_val_nodes=config.data.max_val_nodes,
            train_size=config.data.train_size,
            val_size=config.data.val_size,
            seed=seed,
            verbose=config.run.silent,
        )

    elif config_file == "debug_ColourInteract.yaml":
        graphs_train, graphs_val = get_data_ColourInteract(
            rewirers=rewirers,
            dataset=config.data.dataset,
            device=device,
            c1=config.data.c1,
            c2=config.data.c2,
            num_colours=config.data.num_colours,
            min_train_nodes=config.data.min_train_nodes,
            max_train_nodes=config.data.max_train_nodes,
            max_val_nodes=config.data.max_val_nodes,
            train_size=config.data.train_size,
            val_size=config.data.val_size,
            seed=seed,
            verbose=config.run.silent,
        )

    else:
        raise NotImplementedError

    print(
        f"train targets: {np.mean([g.y.cpu() for g in graphs_train[0]]):.2f} +/- {np.std([g.y.cpu() for g in graphs_train[0]]):.3f}"
    )
    print(
        f"val targets: {np.mean([g.y.cpu() for g in graphs_val[0]]):.2f} +/- {np.std([g.y.cpu() for g in graphs_val[0]]):.3f}"
    )

    dl_train = DataLoader(graphs_train[0], batch_size=config.train.train_batch_size)
    dl_val = DataLoader(graphs_val[0], batch_size=config.train.val_batch_size)

    print(len(graphs_train))
    in_channels = graphs_train[0][0].x.shape[1]
    out_channels = (
        1 if len(graphs_train[0][0].y.shape) == 0 else len(graphs_train[0][0].y.shape)
    )

    print("-------------------")
    print("Training a GIN model without rewiring...")

    # model = GINModel(
    #     in_channels=in_channels,
    #     hidden_channels=config.model.hidden_channels,
    #     num_layers=config.model.num_layers,
    #     out_channels=out_channels,
    #     drop_prob=config.model.drop_prob,
    #     only_original_graph=True,
    # )
    # model.to(device)

    # train_eval_loop(
    #     model,
    #     dl_train,
    #     dl_val,
    #     lr=config.train.lr,
    #     num_epochs=config.train.num_epochs,
    #     print_every=config.train.print_every,
    #     verbose=True,
    #     log_wandb=False,
    # )

    for num, rewirer in enumerate(rewirers):
        print("-------------------")
        print(f"Training a GIN model + interleaved {rewirer}...")

        dl_train = DataLoader(
            graphs_train[num], batch_size=config.train.train_batch_size
        )
        dl_val = DataLoader(graphs_val[num], batch_size=config.train.val_batch_size)

        in_channels = graphs_train[num][0].x.shape[1]
        out_channels = (
            1
            if len(graphs_train[num][0].y.shape) == 0
            else len(graphs_train[num][0].y.shape)
        )

        model = GINModel(
            in_channels=in_channels,
            hidden_channels=config.model.hidden_channels,
            num_layers=config.model.num_layers,
            out_channels=out_channels,
            drop_prob=config.model.drop_prob,
            interleave_diff_graph=True,
        )
        model.to(device)

        end_results = train_eval_loop(
            model,
            dl_train,
            dl_val,
            lr=config.train.lr,
            num_epochs=config.train.num_epochs,
            print_every=config.train.print_every,
            verbose=True,
            log_wandb=False,
        )


def run_experiment(config_fn):
    with open(f"configs/{config_fn}", "r") as f:
        config = EasyDict(yaml.safe_load(f))

    print(config)

    os.environ["WANDB_SILENT"] = str(config.run.silent).lower()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    assert (
        (config.model.approach == "only_original")
        or (config.model.approach == "only_diff")
        or (config.model.approach == "interleave")
    )
    if config.model.approach == "only_original":
        print("No rewiring.")
        num_runs = len(config.data.seeds)
        config.data.rewirers = [None]
    else:
        num_runs = len(config.data.seeds) * len(config.data.rewirers)

    results = {rewirer: [] for rewirer in config.data.rewirers}

    with tqdm(total=num_runs, disable=not config.run.silent) as pbar:
        for seed in config.data.seeds:
            # load data

            if config.data.name == "SalientDists":
                graphs_train, graphs_val = get_data_SalientDists(
                    rewirers=config.data.rewirers,
                    dataset=config.data.dataset,
                    device=device,
                    c1=config.data.c1,
                    c2=config.data.c2,
                    c3=config.data.c3,
                    d=config.data.d,
                    min_train_nodes=config.data.min_train_nodes,
                    max_train_nodes=config.data.max_train_nodes,
                    max_val_nodes=config.data.max_val_nodes,
                    train_size=config.data.train_size,
                    val_size=config.data.val_size,
                    seed=seed,
                    verbose=config.run.silent,
                )

            elif config.data.name == "ColourInteract":
                graphs_train, graphs_val = get_data_ColourInteract(
                    rewirers=config.data.rewirers,
                    dataset=config.data.dataset,
                    device=device,
                    c1=config.data.c1,
                    c2=config.data.c2,
                    num_colours=config.data.num_colours,
                    min_train_nodes=config.data.min_train_nodes,
                    max_train_nodes=config.data.max_train_nodes,
                    max_val_nodes=config.data.max_val_nodes,
                    train_size=config.data.train_size,
                    val_size=config.data.val_size,
                    seed=seed,
                    verbose=config.run.silent,
                )

            print(
                f"train targets: {np.mean([g.y.cpu() for g in graphs_train[0]]):.2f} +/- {np.std([g.y.cpu() for g in graphs_train[0]]):.3f}"
            )

            print(
                f"val targets: {np.mean([g.y.cpu() for g in graphs_val[0]]):.2f} +/- {np.std([g.y.cpu() for g in graphs_val[0]]):.3f}"
            )

            in_channels = graphs_train[0][0].x.shape[1]
            out_channels = (
                1
                if len(graphs_train[0][0].y.shape) == 0
                else len(graphs_train[0][0].y.shape)
            )

            # run experiment for each rewirer

            for ind_rewirer, rewirer in enumerate(config.data.rewirers):
                print(f"Rewirer {rewirer} with seed {seed}")

                dl_train = DataLoader(
                    graphs_train[ind_rewirer], batch_size=config.train.train_batch_size
                )
                dl_val = DataLoader(
                    graphs_val[ind_rewirer], batch_size=config.train.val_batch_size
                )

                wandb.init(
                    project=config.wandb.project,
                    entity=config.wandb.entity,
                    config=config,
                    group=config.wandb.experiment_name
                    + f"-{config.model.approach}-rewired-with-{rewirer}",
                )

                wandb.run.name = (
                    config.wandb.experiment_name
                    + f"-{config.model.approach}-rewired-with-{rewirer}-seed-{seed}"
                )

                model = GINModel(
                    in_channels=in_channels,
                    hidden_channels=config.model.hidden_channels,
                    num_layers=config.model.num_layers,
                    out_channels=out_channels,
                    drop_prob=config.model.drop_prob,
                    only_original_graph=(config.model.approach == "only_original"),
                    interleave_diff_graph=(config.model.approach == "interleave"),
                    only_diff_graph=(config.model.approach == "only_diff"),
                ).to(device)

                final_val_loss = train_eval_loop(
                    model,
                    dl_train,
                    dl_val,
                    lr=config.train.lr,
                    num_epochs=config.train.num_epochs,
                    print_every=config.train.print_every,
                    verbose=not config.run.silent,
                    log_wandb=True,
                )

                results[rewirer].append(final_val_loss)

                wandb.finish()
                pbar.update(1)

    with open("data/results/" + config.wandb.experiment_name + ".json", "w") as f:
        json.dump(results, f)

    pprint(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_fn", help="configuration file name", type=str)
    args = parser.parse_args()

    run_experiment(args.config_fn)

    # quick_run(["aligned_cayley", "cayley", "fully_connected", "interacting_pairs"],
    #   "debug_SalientDists.yaml")
    # quick_run(
    #     ["cayley", "fully_connected", "cayley_clusters"], "debug_ColourInteract.yaml"
    # )


if __name__ == "__main__":
    main()
