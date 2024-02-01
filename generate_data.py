# %%

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader

from data import TanhMix


# %%

def make_dummy():

    dummy_data = Data(
        x = torch.zeros((5,1)),
        y = None,
        edge_attr = None,
        edge_index=torch.tensor([[0,0,0,2],[1,2,3,4]]),
    )

    g = TanhMix(id='dummy', data=dummy_data, seed=0)

    g.draw()
    print(f"{g.num_nodes=} {g.num_edges=} {g.dim_feat=}")
    print(g.x)
    print(g.y)
    print(g.interact_strength)

    print("returning g")
    return g



def make_zinc():

    zinc_graphs = ZINC(
        root = 'data_zinc',
        subset = True,
        split = 'train',
    )

    g = TanhMix(id=123, data=zinc_graphs[123], seed=123)

    g.draw()
    print(f"{g.num_nodes=} {g.num_edges=} {g.dim_feat=}")
    print(g.x)
    print(g.y)
    print(g.interact_strength)



def get_data():
    
    zinc_graphs = ZINC(
        root = 'data_zinc',
        subset = True,
        split = 'train',
    )


    graphs_train = []
    graphs_val = []

    for i in range(200):
        g = TanhMix(id=i, data=zinc_graphs[i], seed=i)
        graphs_train.append(g.to_torch_data())

    for i in range(100,110):
        g = TanhMix(id=i, data=zinc_graphs[i], seed=i)
        graphs_val.append(g.to_torch_data())


    # can't batch differently sized graphs, need to implement
    dl_train = DataLoader(graphs_train, batch_size=1) 
    dl_val = DataLoader(graphs_val, batch_size=1)
    
    return dl_train, dl_val



def main():

    make_dummy()
    # make_zinc()
    # get_data()



if __name__ == "__main__":
    main()
# %%
