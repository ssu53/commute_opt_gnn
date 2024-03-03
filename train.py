import argparse
import json
import os
import random
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        f"Minimum val loss at epoch {min(epoch2valloss, key=epoch2valloss.get)}: {min(epoch2valloss.values())}"
    )

    print(f"Final val loss at epoch {num_epochs}: {epoch2valloss[num_epochs]}")

    if verbose:
        print("Train / Val loss by epoch")
        print(
            "\n".join(
                "{!r}: {:.3f} / {:.3f}".format(
                    epoch, epoch2trainloss[epoch], epoch2valloss[epoch]
                )
                for epoch in epoch2trainloss
            )
        )

    end_results = {"end": val_loss, "best": min(epoch2valloss.values())}

    del optimiser
    del loss_fn

    return end_results


def quick_run(rewirers, config_file="debug_ColourInteract.yaml"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(f"configs/{config_file}", "r") as f:
        config = EasyDict(yaml.safe_load(f))

    pprint(config)

    # Get data
    set_seed(config.data.seed)

    if config.data.name == "SalientDists":
        graphs_train, graphs_val = get_data_SalientDists(
            dataset=config.data.dataset,
            device=device,
            c1=config.data.c1,
            c2=config.data.c2,
            c3=config.data.c3,
            d=config.data.d,
            normalise=config.data.normalise,
            min_train_nodes=config.data.min_train_nodes,
            max_train_nodes=config.data.max_train_nodes,
            min_val_nodes=config.data.min_val_nodes,
            max_val_nodes=config.data.max_val_nodes,
            train_size=config.data.train_size,
            val_size=config.data.val_size,
            verbose=config.run.silent,
        )
    elif config.data.name == "ColourInteract":
        graphs_train, graphs_val = get_data_ColourInteract(
            dataset=config.data.dataset,
            device=device,
            c1=config.data.c1,
            c2=config.data.c2,
            normalise=config.data.normalise,
            num_colours=config.data.num_colours,
            min_train_nodes=config.data.min_train_nodes,
            max_train_nodes=config.data.max_train_nodes,
            min_val_nodes=config.data.min_val_nodes,
            max_val_nodes=config.data.max_val_nodes,
            train_size=config.data.train_size,
            val_size=config.data.val_size,
            verbose=config.run.silent,
        )
    else:
        raise NotImplementedError

    print(
        f"train targets: {np.mean([g.y.cpu() for g in graphs_train]):.2f} +/- {np.std([g.y.cpu() for g in graphs_train]):.3f}"
    )
    print(
        f"val targets: {np.mean([g.y.cpu() for g in graphs_val]):.2f} +/- {np.std([g.y.cpu() for g in graphs_val]):.3f}"
    )

    graphs_train_rewirer = []
    for g in graphs_train:
        g.attach_rewirer(config.data.rewirer)
        graphs_train_rewirer.append(g.to_torch_data().to(device))

    graphs_val_rewirer = []
    for g in graphs_val:
        g.attach_rewirer(config.data.rewirer)
        graphs_val_rewirer.append(g.to_torch_data().to(device))

    for seed in config.model.seeds:
        config.model.seed = seed

    set_seed(config.model.seed)

    dl_train = DataLoader(graphs_train_rewirer, batch_size=config.train.train_batch_size)
    dl_val = DataLoader(graphs_val_rewirer, batch_size=config.train.val_batch_size)

    print(len(graphs_train))
    in_channels = graphs_train[0].x.shape[1]
    out_channels = (
        1 if len(graphs_train[0].y.shape) == 0 else len(graphs_train[0].y.shape)
    )

    dl_train = DataLoader(
        graphs_train, batch_size=config.train.train_batch_size
    )
    dl_val = DataLoader(graphs_val, batch_size=config.train.val_batch_size)

    in_channels = graphs_train[0].x.shape[1]
    out_channels = (
        1
        if len(graphs_train[0].y.shape) == 0
        else len(graphs_train[0].y.shape)
    )

    model = GINModel(
        in_channels=in_channels,
        hidden_channels=config.model.hidden_channels,
        num_layers=config.model.num_layers,
        out_channels=out_channels,
        drop_prob=config.model.drop_prob,
        interleave_diff_graph=True,
        global_pool_aggr=config.model.global_pool_aggr,
        norm=config.model.norm,
    )
    model.to(device)

    end_results = train_eval_loop(
        model,
        dl_train,
        dl_val,
        lr=config.train.lr,
        num_epochs=config.train.num_epochs,
        print_every=config.train.print_every,
        verbose=not config.run.silent,
        log_wandb=False,
    )

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
            global_pool_aggr=config.model.global_pool_aggr,
            norm=config.model.norm
        )
        model.to(device)

        end_results = train_eval_loop(
            model,
            dl_train,
            dl_val,
            lr=config.train.lr,
            num_epochs=config.train.num_epochs,
            print_every=config.train.print_every,
            verbose=not config.run.silent,
            log_wandb=False,
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(config, graphs_train, graphs_val):
    print(config)
    os.environ["WANDB_SILENT"] = str(config.run.silent).lower()
    print(f"Device: {device}")

    assert (
        (config.model.approach == "only_original")
        or (config.model.approach == "only_diff")
        or (config.model.approach == "interleave")
    )

    if config.model.approach == "only_original":
        print("No rewiring.")
        rewirers = [None]
    else:
        rewirers = config.data.rewirers

    results = {rewirer: [] for rewirer in rewirers}

    train_mean = np.mean([g.y.cpu() for g in graphs_train])
    train_std = np.std([g.y.cpu() for g in graphs_train])
    print(f"train targets: {train_mean:.2f} +/- {train_std:.3f}")

    val_mean = np.mean([g.y.cpu() for g in graphs_val])
    val_std = np.std([g.y.cpu() for g in graphs_val])
    print(f"val targets: {val_mean:.2f} +/- {val_std:.3f}")

    in_channels = graphs_train[0].x.shape[1]
    out_channels = (
        1 if len(graphs_train[0].y.shape) == 0 else len(graphs_train[0].y.shape)
    )

    # run experiment for each rewirer

    for rewirer in rewirers:
        set_seed(config.data.seed)

        graphs_train_rewirer = []
        for g in graphs_train:
            g.attach_rewirer(rewirer)
            graphs_train_rewirer.append(g.to_torch_data().to(device))

        graphs_val_rewirer = []
        for g in graphs_val:
            g.attach_rewirer(rewirer)
            graphs_val_rewirer.append(g.to_torch_data().to(device))

        for seed in config.model.seeds:
            config.model.seed = seed

            set_seed(config.model.seed)

            print(f"Rewirer {rewirer} with seed {config.model.seed}")

            dl_train = DataLoader(
                graphs_train_rewirer, batch_size=config.train.train_batch_size
            )
            dl_val = DataLoader(
                graphs_val_rewirer, batch_size=config.train.val_batch_size
            )

            
            if config.data.name == "SalientDists":
                wandb.init(
                    project=config.wandb.project,
                    entity=config.wandb.entity,
                    config=config,
                    group=f"{config.wandb.experiment_name}-{config.model.approach}-{rewirer}-c1-{config.data.c1}-c2-{config.data.c2}-c3-{config.data.c3}",
                )
                wandb.run.name = f"{config.wandb.experiment_name}-{config.model.approach}-{rewirer}-c1-{config.data.c1}-c2-{config.data.c2}-c3-{config.data.c3}-seed-{config.model.seed}"
            
            if config.data.name == "ColourInteract":
                wandb.init(
                    project=config.wandb.project,
                    entity=config.wandb.entity,
                    config=config,
                    group=f"{config.data.dataset}-{config.model.approach}-rewired-with-{rewirer}-c2-{config.data.c2}",
                )
                wandb.run.name = f"{config.data.dataset}-{config.model.approach}-rewired-with-{rewirer}-c2-{config.data.c2}-seed-{config.model.seed}"
            
            
            wandb.log({"train/target_mean": train_mean, "train/target_std": train_std})
            wandb.log({"eval/target_mean": val_mean, "eval/target_std": val_std})

            model = GINModel(
                in_channels=in_channels,
                hidden_channels=config.model.hidden_channels,
                num_layers=config.model.num_layers,
                out_channels=out_channels,
                drop_prob=config.model.drop_prob,
                only_original_graph=(config.model.approach == "only_original"),
                interleave_diff_graph=(config.model.approach == "interleave"),
                only_diff_graph=(config.model.approach == "only_diff"),
                norm=config.model.norm,
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

            del model

            wandb.finish()

    with open("data/results/" + config.wandb.experiment_name + ".json", "w") as f:
        json.dump(results, f)

    pprint(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_fn", help="configuration file name", type=str)
    args = parser.parse_args()

    with open(f"configs/{args.config_fn}", "r") as f:
        config = EasyDict(yaml.safe_load(f))
    print(config)

    for c2 in config.data.c2s:
        set_seed(config.data.seed)
        config.data.c2 = c2
        if config.data.name == "SalientDists":
            graphs_train, graphs_val = get_data_SalientDists(
                dataset=config.data.dataset,
                c1=config.data.c1,
                c2=config.data.c2,
                c3=config.data.c3,
                d=config.data.d,
                normalise=config.data.normalise,
                min_train_nodes=config.data.min_train_nodes,
                max_train_nodes=config.data.max_train_nodes,
                min_val_nodes=config.data.min_val_nodes,
                max_val_nodes=config.data.max_val_nodes,
                train_size=config.data.train_size,
                val_size=config.data.val_size,
                verbose=config.run.silent,
            )
        elif config.data.name == "ColourInteract":
            graphs_train, graphs_val = get_data_ColourInteract(
                dataset=config.data.dataset,
                c1=config.data.c1,
                c2=config.data.c2,
                normalise=config.data.normalise,
                num_colours=config.data.num_colours,
                min_train_nodes=config.data.min_train_nodes,
                max_train_nodes=config.data.max_train_nodes,
                min_val_nodes=config.data.min_val_nodes,
                max_val_nodes=config.data.max_val_nodes,
                train_size=config.data.train_size,
                val_size=config.data.val_size,
                verbose=config.run.silent,
            )
        for approach in config.model.approaches:
            config.model.approach = approach

            run_experiment(config, graphs_train, graphs_val)

    # quick_run(
    #     [
    #         "cayley",
    #         # "fully_connected",
    #         "aligned_cayley",
    #         # "interacting_pairs"
    #     ],
    #   "debug_SalientDists_ZINC.yaml"
    #   )

    # quick_run(
    #     [
    #         # "fully_connected",
    #         "cayley_clusters",
    #         "cayley",
    #     ],
    #     "debug_ColourInteract.yaml"
    # )


if __name__ == "__main__":
    main()
