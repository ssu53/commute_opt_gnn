# %%

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.utils import to_networkx

from generate_data import load_base_graphs

# %%

def get_filtered_graphs(
    all_graphs,
    min_nodes = 30,
    max_nodes = 45,
    d = 5,
):
    
    graphs = []

    for i in range(len(all_graphs)):
        nx_graph = to_networkx(all_graphs[i], to_undirected=True)
        conditions = (
            nx.is_connected(nx_graph)
            and nx_graph.number_of_nodes() > min_nodes
            and nx_graph.number_of_nodes() < max_nodes
            and nx.diameter(nx_graph) > d
        )
        if conditions:
            graphs.append(nx_graph)

    print(f"number of graphs: {len(graphs)}")

    print(
        f"#nodes: {np.mean([g.number_of_nodes() for g in graphs]):.2f} +/- {np.std([g.number_of_nodes() for g in graphs]):.3f}"
    )

    return graphs

# %%

all_graphs_train, all_graphs_val = load_base_graphs('ZINC')

# %%

node_counts_train = [len(dat.x) for dat in all_graphs_train]
node_counts_val = [len(dat.x) for dat in all_graphs_val]

# %%

plt.figure()
plt.hist(node_counts_train, bins=np.arange(0,40,1))
plt.title("number of nodes in train set")
plt.show()

plt.figure()
plt.hist(node_counts_val, bins=np.arange(0,40,1))
plt.title("number of nodes in val set")
plt.show()


# %%

graphs_val = get_filtered_graphs(all_graphs_val, 30, 40)

plt.figure()
plt.hist([g.number_of_nodes() for g in graphs_val], bins=np.arange(0,40,1))
plt.title("number of nodes in val filtered")
plt.show()


# %%
