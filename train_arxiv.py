# %%

import torch

from ogb.nodeproppred import PygNodePropPredDataset
from models import GINModel
from train import count_parameters, set_seed

from easydict import EasyDict
import yaml
import tqdm
import wandb




def get_ogbn_arxiv():
    """
    see data description at https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv
    """

    dataset = PygNodePropPredDataset(name = "ogbn-arxiv", root = "data") 

    NUM_CLASSES = 40

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph = dataset[0] # pyg graph object

    assert len(train_idx) + len(valid_idx) + len(test_idx) == graph.num_nodes
    assert graph.x.size(0) == graph.num_nodes
    assert torch.min(graph.y).item() == 0
    assert torch.max(graph.y).item() == NUM_CLASSES-1

    print(f"train frac {len(train_idx) / graph.num_nodes : .3f}")
    print(f"valid frac {len(valid_idx) / graph.num_nodes : .3f}")
    print(f"test frac {len(test_idx) / graph.num_nodes : .3f}")

    return graph, train_idx, valid_idx, test_idx, NUM_CLASSES



def get_accuracy(logits, y, batch_size = None):
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
            logits_chunk = logits[batch:batch+batch_size]
            y_chunk = y[batch:batch+batch_size]
            y_hat_chunk = torch.argmax(logits_chunk, dim=-1, keepdim=True)
            accs.append((y_chunk == y_hat_chunk).sum().item())

        print("num batches", len(accs))
        acc = sum(accs) / num_nodes
    
    return acc



def train_model(data, model, mask, optimiser, loss_fn = torch.nn.functional.cross_entropy):
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
    if sum(mask).item() == 0: return torch.nan
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
    
    # optimiser and los function
    # -------------------------------

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean") # xent loss for multiclass classification


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
        wandb.log({
            "train/acc": train_acc,
            "val/acc": val_acc,
            # "test/acc": test_acc,
        })


    # training loop
    # -------------------------------

    print("Whole batch gradient descent on one graph...")

    with tqdm.auto.tqdm(range(1,num_epochs+1), unit="e", disable=not verbose) as tepoch:
    # with tqdm(range(1,num_epochs+1), unit="e", disable=not verbose) as tepoch:
        for epoch in tepoch:

            train_loss = train_model(data, model, train_idx, optimiser, loss_fn)

            if log_wandb:
                wandb.log({
                    "train/loss": train_loss
                })

            if epoch % print_every == 0:

                train_acc = eval_model(data, model, train_idx)
                val_acc = eval_model(data, model, val_idx)

                epoch2trainloss[epoch] = train_loss
                epoch2trainacc[epoch] = train_acc
                epoch2valacc[epoch] = val_acc
                
                if log_wandb:
                    wandb.log({
                        "train/acc": train_acc,
                        "val/acc": val_acc,
                    })

                tepoch.set_postfix(train_loss=train_loss, train_acc=train_acc, val_acc=val_acc)
                    

    # final metrics
    # -------------------------------
    
    train_acc = eval_model(data, model, train_idx)
    val_acc = eval_model(data, model, val_idx)
    # test_acc = eval_model(data, model, test_mask)

    epoch = num_epochs
    epoch2trainacc[epoch] = train_acc
    epoch2valacc[epoch] = val_acc

    if log_wandb:
        wandb.log({
            "train/acc": train_acc,
            "val/acc": val_acc,
            # "test/acc": test_acc,
        })

    # print metrics
    # -------------------------------

    print(f"Max val acc at epoch {max(epoch2valacc, key=epoch2valacc.get)}: {max(epoch2valacc.values())}")
    print(f"Final val acc at epoch {num_epochs}: {epoch2valacc[num_epochs]}")

    if verbose:
        print("epoch: train loss | train acc | val acc")
        print(
            "\n".join(
                "{!r}: {:.5f} | {:.3f} | {:.3f}".format(
                    epoch, epoch2trainloss[epoch], epoch2trainacc[epoch], epoch2valacc[epoch],
                )
                for epoch in epoch2trainloss # this doesn't print the 0th epoch before training
            )
        )

    end_results = {
        "end": val_acc,
        "best": min(epoch2valacc.values()),
    }

    del optimiser
    del loss_fn

    return end_results



def main():


    # set device
    # -------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")


    # load config
    # -------------------------------
    with open(f"configs/ogbn-arxiv.yaml", "r") as f:
        config = EasyDict(yaml.safe_load(f))


    # load dataset
    # -------------------------------
    graph, train_idx, valid_idx, test_idx, num_classes = get_ogbn_arxiv()
    graph.to(device)
    config.model.in_channels = graph.x.size(1)
    config.model.out_channels = num_classes

    print(config)

    
    # get moodel
    # -------------------------------

    set_seed(config.model.seed)

    model = GINModel(
        in_channels=config.model.in_channels, 
        hidden_channels=config.model.hidden_channels,
        num_layers=config.model.num_layers,
        out_channels=config.model.out_channels,
        drop_prob=config.model.drop_prob,
        only_original_graph=True,
        global_pool_aggr=None,
        norm=config.model.norm,
    )
    model.to(device)

    count_parameters(model)
    print(model)


    # train
    # -------------------------------

    if config.train.log_wandb:
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            config=config,
            group=f"{config.wandb.experiment_name}-lr-{config.train.lr}"
        )
        wandb.run.name = f"{config.wandb.experiment_name}-seed-{config.model.seed}"
    

    end_results = train_eval_loop(
        model,
        graph,
        train_idx,
        valid_idx,
        lr = config.train.lr,
        num_epochs = config.train.num_epochs,
        print_every = config.train.print_every,
        verbose = config.train.verbose,
        log_wandb = config.train.log_wandb,
    )



if __name__ == "__main__":
    main()