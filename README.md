# Commute-time-optimal graphs for GNNs

## Setup

```
conda create --prefix /path/to/here/commute-opt-gnns-env python=3.11
conda activate /path/to/here/commute-opt-gnns-env

pip install torch --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt
```


## Models

Currently, we only support working with a GIN convolution model: the Graph Isomorphism Network (GIN) introduced here https://arxiv.org/abs/1810.00826. 

Our implementation `MyGINConv` of the convolution (in particular, the definition of the MLP), mimics the definition in the EGP (Wilson + Velikovic) code. `num_layers` of this module, combined with linear input and output projections, make up our `GINModel`. 

All models are compatible with graph batching. 


## Train and evaluate

```
cd src
```

### Synthetic data

For desired values of `i`, `j` and `k`:


```
python train.py --config_fn ../configs/ColourInteract.yaml --c2_over_c1 k 

python train.py --config_fn ../configs/SalientDists.yaml --c1 i --c2 j --c3 k
```

The dataset is loaded with the rewiring choice. 

For the `SalientDists` synthetic data, availble rewirings are: 

* `"cayley"` (BFS-trimmed Cayley expander graph)
* `"interacting_pairs"` (all pairs at distance 1 and `d` that interact with coefficients `c1`, `c2`, as per the definition of this dataset, are connected) 
* `"distance_d_pairs"` (all pairs at `d` that interact coefficient `c2` connected) 
* `"aligned_cayley"` (union of pairs at distance `d` informs the non-random alignment of Cayley expander over base graph)
* `"fully_connected"` (all nodes connected)a

For the `ColourInteract` synthetic data, availble rewirings are: 
* `"cayley"` (BFS-trimmed Cayley expander graph)
* `"cayley_clusters"` (nodes of the same colour form a cayley subgraph, and subgraphs are sparsely connected with a single extra node/cluster) 
* `"unconnected_cayley_clusters"` (nodes of the same colour form a cayley subgraph) 
* `"fully_connected_clusters"` (all nodes of each colour are connected with eachother)
* `"fully_connected"` (all nodes connected)

### `obgn-arxiv`

(optional) Train MLP for mlp-based colours:

```
python train_arxiv_mlp.py 
```

Precompute rewirings:

```
python compute_arxiv_rewire.py
```

Train models:

```
python train_arxiv_gin.py --config_fn ../configs/obgn-arxiv.yaml
```




## To just generate the Cayley Graphs

```python
V = 40 # The number of vertices of the input graph

generator = CayleyGraphGenerator(V) # Instantiate the Cayley graph generator by computing the size of the smallest Cayley graph with at least V nodes
generator.generate_cayley_graph() # Generate the Cayley graph
generator.trim_graph() # Trim the Graph with BFS to have V nodes again
generator.visualize_graph() # Visualize the graph (optional trimmed=False to see graph before trimming)
```