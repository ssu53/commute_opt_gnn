import pickle
import time
import argparse
import einops
import networkx as nx
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn.decomposition import PCA
from tqdm import tqdm

from generate_cayley_graph import CayleyGraphGenerator
from train_arxiv_gin import get_ogbn_arxiv
from train_arxiv_mlp import MLP

import torch.nn.functional as F
from torch_geometric.nn import CorrectAndSmooth
from torch_geometric.utils.convert import from_networkx

import random


import pathlib
path_root = pathlib.Path(__file__).parent.resolve() / ".."



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

def get_colours_from_kmeans_adaptive(feats):

    for n_clusters in range(2, 40):

        print(f"n_clusters = {n_clusters}")

        clusterer = KMeans(n_clusters=n_clusters, random_state=10, verbose=True)
        cluster_labels = clusterer.fit_predict(feats)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters

        silhouette_avg = silhouette_score(feats, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg
        )

    exit()


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
def get_colours_from_mlp_cs(
    graph,
    train_idx,
    valid_idx,
    device="cpu",
    num_correction_layers = 10,
    correction_alpha = 0.8,
    num_smoothing_layers = 10,
    smoothing_alpha = 0.8,
    autoscale = True,
    scale = None,
    ):
    """
    with correct and smooth
    """

    feats = graph.x
    y_all = graph.y.squeeze()
    y_train = graph.y[train_idx]

    model = MLP(
        feats.size(-1), hidden_channels=256, out_channels=40, num_layers=3, dropout=0.5
    ).to(device)

    model.load_state_dict(
        torch.load(
            "../data/models/ogbn-arxiv-mlp-model.pth", map_location=torch.device("cpu")
        )
    )

    out = model(feats)
    y_soft = out.softmax(dim=-1)

    print("Before post-processing")
    preds = y_soft.argmax(dim=-1)
    print(f"train acc: {(preds[train_idx]  == y_all[train_idx]).sum() / train_idx.size(0)}")
    print(f"val acc: {(preds[valid_idx]  == y_all[valid_idx]).sum() / valid_idx.size(0)}")


    print(f"{num_correction_layers=} {correction_alpha=} {num_smoothing_layers=} {smoothing_alpha=} {autoscale=} {scale=}")

    print("Initialising CorrectAndSmooth processor")
    cs_processor = CorrectAndSmooth(
        num_correction_layers = num_correction_layers,
        correction_alpha = correction_alpha,
        num_smoothing_layers = num_smoothing_layers,
        smoothing_alpha = smoothing_alpha,
        autoscale = True,
        scale = None,
    )

    print("Correcting...")
    y_soft = cs_processor.correct(y_soft, y_train, train_idx, graph.edge_index)
    print("Smoothing...")
    y_soft = cs_processor.smooth(y_soft, y_train, train_idx, graph.edge_index)


    print("After correct and smooth")
    preds = y_soft.argmax(dim=-1)
    print(f"train acc: {(preds[train_idx]  == y_all[train_idx]).sum() / train_idx.size(0)}")
    print(f"val acc: {(preds[valid_idx]  == y_all[valid_idx]).sum() / valid_idx.size(0)}")

    return preds



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



def get_colours_from_knn(X, y, train_idx, valid_idx, test_idx, n_neighbs=10, pca_components=None):

    if pca_components is not None:
        pca = PCA(n_components=pca_components)
        X = pca.fit_transform(X)
    
    knn = KNeighborsClassifier(n_neighbs)
    knn.fit(X[train_idx], y[train_idx].squeeze(-1))

    colours = np.ones(y.shape, dtype=int) * -1
    colours[train_idx] = y[train_idx]
    colours[valid_idx] = knn.predict(X[valid_idx]).reshape(-1,1)
    colours[test_idx] = knn.predict(X[test_idx]).reshape(-1,1)

    colours = torch.tensor(colours)

    return colours




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assign_colours_by",
        default="mlp_cs_all",
        help="Method for assigning colours for Cayley clusters",
        type=str,
    )
    args = parser.parse_args()

    assign_colours_by = args.assign_colours_by
    print(f"Computing rewiring according to: {assign_colours_by}")

    graph, train_idx, valid_idx, test_idx, num_classes = get_ogbn_arxiv()
    
    # num_nodes = graph.y.shape[0]
    # proportion_corrupted = 0.5
    # num_corrupted = int(num_nodes * proportion_corrupted)

    # boolean_list = [1] * num_corrupted + [0] * (num_nodes - num_corrupted)
    # random.shuffle(boolean_list)
    # boolean_list = torch.tensor(boolean_list, dtype=torch.bool)

    # corrupted_values = torch.randint(0, num_classes, (num_corrupted,))

    # corrupted_y = graph.y.squeeze().detach().clone()
    # corrupted_y[boolean_list] = corrupted_values

    # print(sum(corrupted_y == graph.y.squeeze()) / num_nodes)

    # rewire_edge_index = get_cayley_clusters_rewiring(
    #     corrupted_y,
    #     allowable_idx=torch.tensor(range(graph.num_nodes)),
    #     num_colours=num_classes,
    # )

    # with open("../data/arxiv-rewirings/arxiv_rewire_by_corrupted_0_5_class_all", "wb") as f:
    #     pickle.dump(rewire_edge_index, f)

    # check_valid(
    #     rewire_edge_index, corrupted_y, torch.tensor(range(graph.num_nodes))
    # )

    # save_dict = {
    #     "graph": graph,
    #     "train_idx": train_idx,
    #     "valid_idx": valid_idx,
    #     "test_idx": test_idx,
    #     "num_classes": num_classes,
    #     "corrupted_y": corrupted_y.unsqueeze(1),
    # }

    # with open("../data/ogbn_arxiv/corrupted_0_5_data.pkl", "wb") as f:
    #     pickle.dump(save_dict, f)

    if assign_colours_by == "none":
        # with no colur assignments, construct Cayley expander over entire graph
        # this is agnostic to graph features and labels

        num_nodes = graph.num_nodes

        if num_nodes < 2:
            raise ValueError(
                "Cayley graph requires at least 2 nodes, but got only {}.".format(
                    num_nodes
                )
            )

        time_start = time.time()
        print("Instantiating CayleyGraphGenerator")
        cg_gen = CayleyGraphGenerator(num_nodes)
        print("Generating...")
        cg_gen.generate_cayley_graph()
        print("Trimming...")
        cg_gen.trim_graph()
        print("Done!")
        time_end = time.time()
        print(f"Time elapsed {time_end-time_start:.3f} s")

        rewired_graph = nx.relabel_nodes(
            cg_gen.G_trimmed, dict(zip(cg_gen.G_trimmed, range(num_nodes)))
        )

        rewire_edge_index = from_networkx(rewired_graph).edge_index

        with open("../data/arxiv-rewirings/arxiv_rewire_by_cayley", "wb") as f:
            pickle.dump(rewire_edge_index, f)
        
        return
    
            
    if assign_colours_by == "by_class_train":
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

        return


    if assign_colours_by == "by_class_all":
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

        return
    
    if assign_colours_by == "by_kmeans_all":

        normalized_features = graph.x / torch.norm(graph.x, p=2, dim=1, keepdim=True)

        # k_means_colours = get_colours_from_kmeans(normalized_features.numpy(), num_classes)
        k_means_colours = get_colours_from_kmeans_adaptive(normalized_features.numpy())

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

        return

    if assign_colours_by == "enriched-kmeans":

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

        return

    
    if assign_colours_by == "mlp_all":

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

        return

    if assign_colours_by == "mlp_cs_all":

        mlp_colours = get_colours_from_mlp_cs(graph, train_idx, valid_idx)

        assert mlp_colours.size(0) == graph.num_nodes

        rewire_edge_index = get_cayley_clusters_rewiring(
            mlp_colours,
            allowable_idx=torch.tensor(range(graph.num_nodes)),
            num_colours=num_classes,
        )

        with open("../data/arxiv-rewirings/arxiv_rewire_by_mlp_cs_all", "wb") as f:
            pickle.dump(rewire_edge_index, f)

        check_valid(rewire_edge_index, mlp_colours, torch.tensor(range(graph.num_nodes)))

        return
    
    if assign_colours_by == "mlp_feats_all":

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

        return
    
        
    if assign_colours_by == "knn":

        features = graph.x
        features = features / torch.norm(features, p=2, dim=1, keepdim=True)
        colours = get_colours_from_knn(features, graph.y, train_idx, valid_idx, test_idx)

        assert colours.shape == graph.y.shape
        assert torch.all(colours >= 0)
        assert torch.all(colours < num_classes)
        
        print("KNN classification acc")
        print("train", torch.sum(colours[train_idx] == graph.y[train_idx]).item() / len(train_idx))
        print("valid", torch.sum(colours[valid_idx] == graph.y[valid_idx]).item() / len(valid_idx))
        print("test", torch.sum(colours[test_idx] == graph.y[test_idx]).item() / len(test_idx))

        rewire_edge_index = get_cayley_clusters_rewiring(
            colours.squeeze(),
            allowable_idx=torch.tensor(range(graph.num_nodes)),
            num_colours=num_classes,
        )

        with open("../data/arxiv-rewirings/arxiv_rewire_by_knn", "wb") as f:
            pickle.dump(rewire_edge_index, f)

        check_valid(
            rewire_edge_index, colours, torch.tensor(range(graph.num_nodes))
        )
        
        return
    
    if assign_colours_by == "knn_mlp_feats":


        model = MLP(
            graph.x.size(-1), hidden_channels=256, out_channels=40, num_layers=3, dropout=0.5
        )

        model.load_state_dict(
            torch.load(
                path_root / "data/models/ogbn-arxiv-mlp-model.pth", map_location=torch.device("cpu")
            )
        )

        out, features = model(graph.x, return_feats=True)
        features = features / torch.norm(features, p=2, dim=1, keepdim=True)
        features = features.detach().numpy()
        colours = get_colours_from_knn(features, graph.y, train_idx, valid_idx, test_idx)

        assert colours.shape == graph.y.shape
        assert torch.all(colours >= 0)
        assert torch.all(colours < num_classes)

        print("KNN classification acc")
        print("train", torch.sum(colours[train_idx] == graph.y[train_idx]).item() / len(train_idx))
        print("valid", torch.sum(colours[valid_idx] == graph.y[valid_idx]).item() / len(valid_idx))
        print("test", torch.sum(colours[test_idx] == graph.y[test_idx]).item() / len(test_idx))

        rewire_edge_index = get_cayley_clusters_rewiring(
            colours.squeeze(),
            allowable_idx=torch.tensor(range(graph.num_nodes)),
            num_colours=num_classes,
        )

        with open(path_root / "data/arxiv-rewirings/arxiv_rewire_by_knn_mlp_feats", "wb") as f:
            pickle.dump(rewire_edge_index, f)

        check_valid(
            rewire_edge_index, colours, torch.tensor(range(graph.num_nodes))
        )

        return

    raise NotImplementedError
        

if __name__ == "__main__":
    main()
