import pickle

import einops
import networkx as nx
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

from generate_cayley_graph import CayleyGraphGenerator
from train_arxiv_gin import get_ogbn_arxiv
from train_arxiv_mlp import MLP

import torch.nn.functional as F


def get_cayley_clusters_rewiring(colours, allowable_idx, num_colours: int):
    """
    num_colours are the number of colours in label, assumed to be indexed from 0 to num_colours-1
    labels < 0 are ignored
    """

    assert colours.ndim == 1

    allowable_idx_set = set(allowable_idx.tolist())
    colours_allowable = colours[allowable_idx]
    map_allowable_to_all = {i: val for i, val in enumerate(allowable_idx.tolist())}

    for i in range(colours_allowable.size(0)):
        assert colours_allowable[i].item() == colours[map_allowable_to_all[i]].item()

    graph = nx.Graph()

    for colour in tqdm(range(num_colours), desc="Generating Cayley clusters"):

        num_nodes = (colours_allowable == colour).sum().item()

        print(colour, num_nodes)

        if num_nodes == 0:
            print("Concerning...")
            continue

        cg_gen = CayleyGraphGenerator(num_nodes)
        cg_gen.generate_cayley_graph()
        cg_gen.trim_graph()

        mapping = {
            org_idx: np.where(colours_allowable == colour)[0][col_idx]
            for org_idx, col_idx in zip(cg_gen.G_trimmed, range(num_nodes))
        }

        remapped_graph = nx.relabel_nodes(cg_gen.G_trimmed, mapping)

        graph = nx.union(graph, remapped_graph)

    # check valid!
    for edge in graph.edges:
        assert colours_allowable[edge[0]] == colours_allowable[edge[1]]
        assert edge[0] < len(colours_allowable)
        assert edge[1] < len(colours_allowable)

    # use original indices, not just indices into allowable
    graph_relabelled = nx.relabel_nodes(graph, map_allowable_to_all)

    # check valid!
    for edge in graph_relabelled.edges:
        assert colours[edge[0]] == colours[edge[1]]
        assert edge[0] in allowable_idx_set, f"{edge}"
        assert edge[1] in allowable_idx_set, f"{edge}"

    rewire_edge_index = einops.rearrange(
        torch.tensor(list(graph_relabelled.edges)),
        "e n -> n e",
    )

    # concatenate the reverse edges to ensure graph is undirected
    rewire_edge_index = torch.cat(
        (rewire_edge_index, rewire_edge_index.flip(dims=(0,))), dim=-1
    )

    return rewire_edge_index


def check_valid(rewire_edge_index, colours, allowable_idx):

    allowable_idx_set = set(allowable_idx.tolist())

    edges = [
        (rewire_edge_index[0, i].item(), rewire_edge_index[1, i].item())
        for i in range(rewire_edge_index.size(1))
    ]
    edges = set(edges)

    for edge in edges:
        node_a, node_b = edge
        assert (node_b, node_a) in edges, f"{edge}"
        assert colours[node_a].item() == colours[node_b].item()
        assert node_a in allowable_idx_set
        assert node_b in allowable_idx_set

    print("Passed basic checks!")


def get_colours_from_kmeans(feats, num_classes):

    kmeans = KMeans(n_clusters=num_classes, n_init=5).fit(feats)

    return torch.from_numpy(kmeans.labels_.astype(np.int64))


@torch.no_grad()
def get_colours_from_mlp(feats, device="cpu"):

    model = MLP(
        feats.size(-1), hidden_channels=256, out_channels=40, num_layers=3, dropout=0.5
    ).to(device)

    model.load_state_dict(
        torch.load(
            "../data/models/ogbn-arxiv-mlp-model.pth", map_location=torch.device("cpu")
        )
    )

    out = model(feats)

    print(out.argmax(dim=-1, keepdim=False).shape)

    return out.argmax(dim=-1, keepdim=False)

@torch.no_grad()
def get_colours_from_mlp_feats(feats, device="cpu"):

    model = MLP(
        feats.size(-1), hidden_channels=256, out_channels=40, num_layers=3, dropout=0.5
    ).to(device)

    model.load_state_dict(
        torch.load(
            "../data/models/ogbn-arxiv-mlp-model.pth", map_location=torch.device("cpu")
        )
    )

    out, mlp_feats = model(feats, return_feats=True)

    kmeans = KMeans(n_clusters=40, n_init=1).fit(mlp_feats)

    return torch.from_numpy(kmeans.labels_.astype(np.int64))


def main():

    graph, train_idx, valid_idx, test_idx, num_classes = get_ogbn_arxiv()

    # ------------------------------
    # Cayley clusters rewire edge index on only the train nodes
    # prevent leakage from val and test set labels into train

    rewire_edge_index = get_cayley_clusters_rewiring(
        graph.y.squeeze(),
        allowable_idx=train_idx,
        num_colours=num_classes,
    )

    with open("../data/arxiv-rewirings/arxiv_rewire_by_class_train_only", "wb") as f:
        pickle.dump(rewire_edge_index, f)

    check_valid(rewire_edge_index, graph.y.squeeze(), train_idx)

    # ------------------------------
    # Cayley clusters rewire edge index on all nodes, irrespective of split
    # this is sensible only in the limit of perfect priors that are equivalent to knowing the label

    rewire_edge_index = get_cayley_clusters_rewiring(
        graph.y.squeeze(),
        allowable_idx=torch.tensor(range(graph.num_nodes)),
        num_colours=num_classes,
    )

    with open("../data/arxiv-rewirings/arxiv_rewire_by_class_all", "wb") as f:
        pickle.dump(rewire_edge_index, f)

    check_valid(
        rewire_edge_index, graph.y.squeeze(), torch.tensor(range(graph.num_nodes))
    )

    # ------------------------------

    normalized_features = graph.x / torch.norm(graph.x, p=2, dim=1, keepdim=True)

    k_means_colours = get_colours_from_kmeans(normalized_features.numpy(), num_classes)

    assert k_means_colours.size(0) == graph.num_nodes

    rewire_edge_index = get_cayley_clusters_rewiring(
        k_means_colours,
        allowable_idx=torch.tensor(range(graph.num_nodes)),
        num_colours=num_classes,
    )

    with open("../data/arxiv-rewirings/arxiv_rewire_by_kmeans_all", "wb") as f:
        pickle.dump(rewire_edge_index, f)

    check_valid(
        rewire_edge_index, k_means_colours, torch.tensor(range(graph.num_nodes))
    )

    # ------------------------------

    one_hot_train = F.one_hot(graph.y.squeeze()[train_idx]).float()
    avg_class = torch.mean(one_hot_train, dim=0)

    enriched_features = torch.zeros((graph.x.shape[0], graph.x.shape[1]+40))

    enriched_features[train_idx] = torch.cat((graph.x[train_idx], one_hot_train), dim=1)
    enriched_features[valid_idx] = torch.cat((graph.x[valid_idx], avg_class.repeat(len(valid_idx), 1)), dim=1)
    enriched_features[test_idx] = torch.cat((graph.x[test_idx], avg_class.repeat(len(test_idx), 1)), dim=1)

    normalized_enriched_features = enriched_features / torch.norm(enriched_features, p=2, dim=1, keepdim=True)

    k_means_colours = get_colours_from_kmeans(normalized_enriched_features.numpy(), num_classes)

    assert k_means_colours.size(0) == graph.num_nodes

    rewire_edge_index = get_cayley_clusters_rewiring(
        k_means_colours,
        allowable_idx=torch.tensor(range(graph.num_nodes)),
        num_colours=num_classes,
    )

    with open("../data/arxiv-rewirings/arxiv_rewire_by_enriched-kmeans_all", "wb") as f:
        pickle.dump(rewire_edge_index, f)

    check_valid(
        rewire_edge_index, k_means_colours, torch.tensor(range(graph.num_nodes))
    )

    # ------------------------------

    mlp_colours = get_colours_from_mlp(graph.x)

    assert mlp_colours.size(0) == graph.num_nodes

    rewire_edge_index = get_cayley_clusters_rewiring(
        mlp_colours,
        allowable_idx=torch.tensor(range(graph.num_nodes)),
        num_colours=num_classes,
    )

    with open("../data/arxiv-rewirings/arxiv_rewire_by_mlp_all", "wb") as f:
        pickle.dump(rewire_edge_index, f)

    check_valid(rewire_edge_index, mlp_colours, torch.tensor(range(graph.num_nodes)))

    # ------------------------------

    mlp_colours = get_colours_from_mlp_feats(graph.x)

    assert mlp_colours.size(0) == graph.num_nodes

    rewire_edge_index = get_cayley_clusters_rewiring(
        mlp_colours,
        allowable_idx=torch.tensor(range(graph.num_nodes)),
        num_colours=num_classes,
    )

    with open("../data/arxiv-rewirings/arxiv_rewire_by_mlp_feats_all", "wb") as f:
        pickle.dump(rewire_edge_index, f)

    check_valid(rewire_edge_index, mlp_colours, torch.tensor(range(graph.num_nodes)))


if __name__ == "__main__":
    main()
