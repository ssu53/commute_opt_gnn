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


def get_data(device=None, rewirer=None, train_size=500, val_size=100):
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

    for i in tqdm(range(train_size)):
        g = ExpMix(id=i, data=zinc_graphs_train[i], seed=i, rewirer=rewirer)
        graphs_train.append(g.to_torch_data().to(device))

    for i in tqdm(range(val_size)):
        g = ExpMix(id=i, data=zinc_graphs_val[i], seed=i, rewirer=rewirer)
        graphs_val.append(g.to_torch_data().to(device))

    return graphs_train, graphs_val
