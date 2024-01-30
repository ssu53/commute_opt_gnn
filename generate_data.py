# %%

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import ZINC

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



def main():

    make_dummy()
    # make_zinc()



if __name__ == "__main__":
    main()
# %%
