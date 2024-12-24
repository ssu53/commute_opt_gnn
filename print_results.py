#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pprint import pprint

import numpy as np
import yaml
from easydict import EasyDict
from tqdm import tqdm

import wandb


# # SalientDists

# In[35]:


api = wandb.Api()

with open("configs/SalientDists.yaml", "r") as file:
    config = EasyDict(yaml.safe_load(file))

c1 = 1.0
c2s = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]
# c3 = 0.0
# c3 = 0.1
c3 = 0.2


# In[36]:


runs = api.runs(f"{config.wandb.entity}/{config.wandb.project}")
print(f"{config.wandb.entity}/{config.wandb.project}")

all_run_names = []

for approach in tqdm(config.model.approaches):
    if approach == "only_original":
        rewirers = [None]
    else:
        rewirers = config.data.rewirers
    for c2 in c2s:
        for rewirer in rewirers:
            statistics = []
            for seed in config.model.seeds:
                
                run_name = f"-{approach}-{rewirer}-c1-{c1}-c2-{c2}-c3-{c3}-seed-{seed}"

                all_run_names.append(run_name)


# In[37]:


all_run_statistics = {}

for run in tqdm(runs):  
    if run.name in all_run_names:
        run_stats = run.history(keys=["eval/loss"], pandas=False)
        run_stats = [stats["eval/loss"] for stats in run_stats]
        all_run_statistics[run.name] = run_stats


# In[38]:


import pickle
with open(f"eval_loss_SalientDists_c3-{c3}", "wb") as f:
    pickle.dump(all_run_statistics, f)


# In[39]:


results = {run: {} for run in ["only_original_None", "interleave_cayley", "interleave_aligned_cayley", "interleave_distance_d_pairs", "interleave_fully_connected"]}

all_run_names = []

for approach in tqdm(config.model.approaches):
    if approach == "only_original":
        rewirers = [None]
    else:
        rewirers = config.data.rewirers
    for c2 in c2s:
        for rewirer in rewirers:
            statistics = []
            for seed in config.model.seeds:
                
                run_name = f"-{approach}-{rewirer}-c1-{c1}-c2-{c2}-c3-{c3}-seed-{seed}"

                statistics.append(all_run_statistics[run_name])

            statistics = np.array(statistics)
            
            mins = statistics[:, -1]

            results[f"{approach}_{rewirer}"][c2] = mins


# In[40]:


import copy

processed_results = copy.deepcopy(results)

for c2 in c2s:

    for rewirer in results:
        normalised_results = []
        
        for seed_idx in range(len(config.model.seeds)):
            plain_gin = results["only_original_None"][c2][seed_idx]
            
            normalised_results.append(processed_results[rewirer][c2][seed_idx] / plain_gin)
            
        processed_results[rewirer][c2] = normalised_results


# In[41]:


pprint(processed_results)


# In[42]:


for rewirer in results:
    for c2 in c2s:
        processed_results[rewirer][c2] = (np.median(processed_results[rewirer][c2]), np.mean(processed_results[rewirer][c2]), np.std(processed_results[rewirer][c2]))


# In[49]:


num_strings = {rewirer: {} for rewirer in results}

for rewirer in results:
    print(f"Rewirer: {rewirer}")
    middle_string = ""
    below_string = ""
    above_string = ""

    for c2 in c2s:
        median, mean, std = processed_results[rewirer][c2]
        below_string += f"({c2:.4f},{max(mean-std, 0.0001):.4f})" # avoid mean-std < 0, which causes downstream plotting on log scale problems
        above_string += f"({c2:.4f},{(mean+std):.4f})"
        middle_string += f"({c2:.4f},{mean:.4f})"

    num_strings[rewirer]["middle"] = middle_string
    num_strings[rewirer]["below"] = below_string
    num_strings[rewirer]["above"] = above_string


# In[50]:


num_strings[rewirer]


# In[51]:


plot = r"""\begin{tikzpicture}
\begin{axis}[
    xmin=0.01, xmax=100,
    ymin=0, ymax=2,
    ymajorgrids=true,
    xlabel=$c_2/c_1$,
    ylabel=$\textrm{loss}/\textrm{loss}_{\textrm{Plain GIN}}$,
    grid style=dashed,
    xmode=log,
    xtick={0.01, 0.1,1,10, 100}, % Specify the positions of the ticks
    xticklabels={$0.01$, $0.1$, $1$, $10$, $100$}, % Specify the labels for the ticks
    ytick={0.0, 0.5, 1.0,1.5,2.0}, % Specify the positions of the ticks
    yticklabels={-, $0.5$, $1.0$, $1.5$, $2.0$}, % Specify the labels for the ticks
    legend pos=north east,
    width=0.9\textwidth,
]
"""

colours = ["black", "blue", "red", "green", "orange", "purple", "brown"]
shading = ["gray", "blue", "red", "green", "orange", "purple", "brown"]

for idx, rewirer in enumerate(results):
    plot += r"\addplot[color=" + colours[idx] + ", ultra thick, mark=ball] coordinates{" + num_strings[rewirer]["middle"] + "};\n\n"

for idx, rewirer in enumerate(results):
    print(rewirer)
    plot += r"\addplot[name path=" + rewirer + "_top,color=" + shading[idx] + r"!70,draw=none] coordinates {" + num_strings[rewirer]["above"] + "};\n\n"
    plot += r"\addplot[name path=" + rewirer + "_down,color=" + shading[idx] + r"!70,draw=none] coordinates {" + num_strings[rewirer]["below"] + "};\n\n"
    # \addplot[gray!50,fill opacity=0.1] fill between[of=gin_top and gin_down];
    plot += r"\addplot[" + shading[idx] + "!50,fill opacity=0.1] fill between[of=" + rewirer + r"_top and " + rewirer + "_down];\n\n"

plot += r"""
\legend{LEGEND ITEM 0, LEGEND ITEM 1, LEGEND ITEM 2, LEGEND ITEM 3, LEGEND ITEM 4}
\end{axis}
\end{tikzpicture}
"""


# In[52]:


print(plot)


# In[ ]:





# # colourinteract

# In[2]:


api = wandb.Api()

with open("configs/ColourInteract.yaml", "r") as file:
    config = EasyDict(yaml.safe_load(file))

c2_over_c1s = [0.01, 0.1, 1.0, 10.0, 100.0]


# In[3]:


runs = api.runs(f"{config.wandb.entity}/{config.wandb.project}")
print(f"{config.wandb.entity}/{config.wandb.project}")

all_run_names = []

for approach in tqdm(config.model.approaches):
    if approach == "only_original":
        rewirers = [None]
    else:
        rewirers = config.data.rewirers
    for c2_over_c1 in c2_over_c1s:
        for rewirer in rewirers:
            statistics = []
            for seed in config.model.seeds:
                
                run_name = f"{config.data.dataset}-{approach}-rewired-with-{rewirer}-c2/c1-{c2_over_c1}-seed-{seed}"

                all_run_names.append(run_name)


# In[4]:


all_run_statistics = {}

for run in tqdm(runs):  
    if run.name in all_run_names:
        run_stats = run.history(keys=["eval/loss"], pandas=False)
        run_stats = [stats["eval/loss"] for stats in run_stats]
        all_run_statistics[run.name] = run_stats


# In[5]:


results = {run: {} for run in ["only_original_None", "interleave_cayley", "interleave_unconnected_cayley_clusters", "interleave_fully_connected_clusters", "interleave_fully_connected"]}

all_run_names = []

for approach in tqdm(config.model.approaches):
    if approach == "only_original":
        rewirers = [None]
    else:
        rewirers = config.data.rewirers
    for c2_over_c1 in c2_over_c1s:
        for rewirer in rewirers:
            statistics = []
            for seed in config.model.seeds:
                
                run_name = f"{config.data.dataset}-{approach}-rewired-with-{rewirer}-c2/c1-{c2_over_c1}-seed-{seed}"

                statistics.append(all_run_statistics[run_name])

            statistics = np.array(statistics)
            
            mins = statistics[:, -1]

            results[f"{approach}_{rewirer}"][c2_over_c1] = mins


# In[6]:


import copy

processed_results = copy.deepcopy(results)

for c2_over_c1 in c2_over_c1s:

    for rewirer in results:
        normalised_results = []
        
        for seed_idx in range(len(config.model.seeds)):
            plain_gin = results["only_original_None"][c2_over_c1][seed_idx]
            
            normalised_results.append(processed_results[rewirer][c2_over_c1][seed_idx] / plain_gin)
            
        processed_results[rewirer][c2_over_c1] = normalised_results


# In[7]:


pprint(processed_results)


# In[8]:


for rewirer in results:
    for c2_over_c1 in c2_over_c1s:
        processed_results[rewirer][c2_over_c1] = (np.median(processed_results[rewirer][c2_over_c1]), np.mean(processed_results[rewirer][c2_over_c1]), np.std(processed_results[rewirer][c2_over_c1]))


# In[9]:


pprint(processed_results)


# In[10]:


num_strings = {rewirer: {} for rewirer in results}

for rewirer in results:
    print(f"Rewirer: {rewirer}")
    middle_string = ""
    below_string = ""
    above_string = ""

    for c2_over_c1 in c2_over_c1s:
        # if c2_over_c1 == 0.5:
        #     print(rewirer)
        #     print(processed_results[rewirer][c2_over_c1])
        median, mean, std = processed_results[rewirer][c2_over_c1]
        below_string += f"({c2_over_c1:.4f},{(mean-std):.4f})"
        above_string += f"({c2_over_c1:.4f},{(mean+std):.4f})"
        middle_string += f"({c2_over_c1:.4f},{mean:.4f})"

    num_strings[rewirer]["middle"] = middle_string
    num_strings[rewirer]["below"] = below_string
    num_strings[rewirer]["above"] = above_string


# In[12]:


plot = r"""\begin{tikzpicture}
\begin{axis}[
    xmin=0.01, xmax=100,
    ymin=0, ymax=2,
    ymajorgrids=true,
    xlabel=$c_2/c_1$,
    ylabel=$\textrm{loss}/\textrm{loss}_{\textrm{Plain GIN}}$,
    grid style=dashed,
    xmode=log,
    xtick={0.01, 0.1,1,10, 100}, % Specify the positions of the ticks
    xticklabels={$0.01$, $0.1$, $1$, $10$, $100$}, % Specify the labels for the ticks
    ytick={0.0, 0.5, 1.0,1.5,2.0}, % Specify the positions of the ticks
    yticklabels={-, $0.5$, $1.0$, $1.5$, $2.0$}, % Specify the labels for the ticks
    legend pos=north east,
    width=0.9\textwidth,
]
"""

colours = ["black", "blue", "red", "green", "orange", "purple", "brown"]
shading = ["gray", "blue", "red", "green", "orange", "purple", "brown"]

for idx, rewirer in enumerate(results):
    plot += r"\addplot[color=" + colours[idx] + ", ultra thick, mark=ball] coordinates{" + num_strings[rewirer]["middle"] + "};\n\n"

for idx, rewirer in enumerate(results):
    print(rewirer)
    plot += r"\addplot[name path=" + rewirer + "_top,color=" + shading[idx] + r"!70,dashed] coordinates {" + num_strings[rewirer]["above"] + "};\n\n"
    plot += r"\addplot[name path=" + rewirer + "_down,color=" + shading[idx] + r"!70,dashed] coordinates {" + num_strings[rewirer]["below"] + "};\n\n"
    # \addplot[gray!50,fill opacity=0.1] fill between[of=gin_top and gin_down];
    plot += r"\addplot[" + shading[idx] + "!50,fill opacity=0.1] fill between[of=" + rewirer + r"_top and " + rewirer + "_down];\n\n"

plot += r"""
\legend{Plain GIN, GIN+Cayley, GIN+Unconnected Cayley Clusters, GIN+Fully Connected}
\end{axis}
\end{tikzpicture}
"""


# In[13]:


print(plot)


# In[ ]:




