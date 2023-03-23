#%%
import networkx as nx
import numpy as np
from graph import Graph
import random
import matplotlib.pyplot as plt

plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.major.size'] = 50

#%%


#Create random graph and assign random weights (with a seed)
nodes = 6
G = nx.erdos_renyi_graph(nodes, 0.7, seed = 5 ) 
random.seed(42)
for (u, v) in G.edges():
    G.edges[u,v]['weight'] = random.uniform(0.01,0.02)


sample_sizes = np.linspace(1, 30, 20)
int_ns = [int(x) for x in sample_sizes]


cut = 10
k = 3
#%%
maxima, errors = Graph().greedy_search2(G, k, n_range= int_ns, repetitions = 40)

#%%
code = 6
#%%

np.save(f'greedy_search_modes,code={code}', maxima)
np.save(f'greedy_search_errors,code = {code}', errors)
#%%
plt.plot(sample_sizes, maxima, 'x-', label = 'greedy search',)
plt.xlabel('Samples')
plt.ylabel('Density')
plt.axhline(y = maxima[-1], color = 'grey', linestyle = '--', label = 'max density')
plt.legend()

#%% Uniform search

uniform_maxs = []
uniform_errors = []
for i in sample_sizes:
    maximum,err = Graph().uniform_search2(G, k, int(i))
    uniform_maxs.append(maximum)
    uniform_errors.append(err)
#%%
np.save(f'uniform_search_modes,code={code}', uniform_maxs)
np.save(f'uniform_search_errors,code={code}', uniform_errors)
#%%


plt.title(f'k={k},nodes = 6')
plt.plot(sample_sizes, uniform_maxs, '.-', label = 'Uniform search',)
plt.errorbar(sample_sizes, maxima, yerr = errors, label=f' Greedy search error', capsize = 3,color = 'purple',linestyle = '')
plt.plot(sample_sizes, maxima, '.-', label = 'Greedy', color = 'purple')
plt.axhline(y = maxima[-1], color = 'grey', linestyle = '--', label = 'max density')
plt.xlabel('Samples')
plt.ylabel('Density')
# plt.axhline(y = uniform_maxs[-1], color = 'grey', linestyle = '--', label = 'max density')
plt.legend()


#%%



with plt.style.context(['science']):
  
    plt.figure(figsize=[8, 6])
    # plt.title('Distinguishability model', fontsize=22)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.title(f'k={k},nodes = 6')
    plt.plot(sample_sizes, uniform_maxs, '.-', label = 'Uniform search',)
    plt.errorbar(sample_sizes, maxima, yerr = errors, label=f' Greedy search error', capsize = 3,color = 'purple',linestyle = '')
    plt.plot(sample_sizes, maxima, '.-', label = 'Greedy', color = 'purple')
    plt.axhline(y = maxima[-1], color = 'grey', linestyle = '--', label = 'max density')
    plt.xlabel('Samples',fontsize = 20)
    plt.ylabel('Density',fontsize = 20)
    # plt.axhline(y = uniform_maxs[-1], color = 'grey', linestyle = '--', label = 'max density')
    plt.legend(fontsize=20)
    plt.savefig(f'greedy_graph_search_code={code}', dpi=600)


#%%
code = 7
maxima_greedy = np.load(f'greedy_search_modes,code={code}.npy',allow_pickle=True)
greedy_error = np.load(f'greedy_search_errors,code = {code}.npy', allow_pickle = True)

uniform_errors = np.load(f'uniform_search_errors,code={code}.npy', allow_pickle= True)
uniform_maxs = np.load(f'uniform_search_modes,code={code}.npy', allow_pickle= True)



with plt.style.context(['science']):
  
    plt.figure(figsize=[8, 6])
    # plt.title('Distinguishability model', fontsize=22)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.title(f'k={k},nodes = 6')
    plt.fill_between(sample_sizes, uniform_maxs + uniform_errors/2, uniform_maxs - uniform_errors/2, color = 'paleturquoise', label = f'Greedy search error')

    plt.plot(sample_sizes, uniform_maxs, '.-', label = 'Uniform search',)
    plt.fill_between(sample_sizes, maxima_greedy + greedy_error/2, maxima_greedy - greedy_error/2, color = 'mediumpurple', label = f'Greedy search error')
    # plt.errorbar(sample_sizes, maxima_greedy, yerr = greedy_error, label=f' Greedy search error', capsize = 3,color = 'purple',linestyle = '')
    plt.plot(sample_sizes, maxima_greedy, '.-', label = 'Greedy', color = 'indigo')
    # plt.errorbar(sample_sizes, uniform_maxs, yerr = uniform_errors, label=f' Uniform search error', capsize = 3,color = 'blue',linestyle = '')
   
    plt.xlabel('Samples',fontsize = 20)
    plt.ylabel('Density',fontsize = 20)
    plt.axhline(y = maxima_greedy[-1], color = 'black', linestyle = '--', label = 'max density')
    # plt.axhline(y = uniform_maxs[-1], color = 'grey', linestyle = '--', label = 'max density')
    plt.legend(fontsize=20)
    plt.savefig(f'greedy_graph_search_code={code}', dpi=600)
    

#%%

with plt.style.context(['science']):
    plt.figure(figsize=(10,8))

    pos = nx.kamada_kawai_layout(G)
    node_options = {'node_color': 'navy', 'node_size':150}
    edge_options = {'width': 2, 'alpha': 0.7, 'edge_color': 'black'}

    nx.draw_networkx_nodes(G, pos, **node_options)
    nx.draw_networkx_edges(G, pos, **edge_options)

    edge_labels = nx.get_edge_attributes(G, "weight")
    edge_labels_rounded = {k: round(v, 4) for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels_rounded)
    plt.savefig('graph_diagram', dpi=600)
    plt.show()


#%%
pos = nx.spring_layout(G, seed=63)  # Seed layout for reproducibility
colors = range(20)
options = {
    "node_color": "#A0CBE2",
    "edge_color": colors,
    "width": 4,
    "edge_cmap": plt.cm.Blues,
    "with_labels": False,
}
nx.draw(G, pos, **options)
plt.show()

#%%

G = nx.Graph()

G.add_edge("a", "b", weight=0.6)
G.add_edge("a", "c", weight=0.2)
G.add_edge("c", "d", weight=0.1)
G.add_edge("c", "e", weight=0.7)
G.add_edge("c", "f", weight=0.9)
G.add_edge("a", "d", weight=0.3)

elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.5]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]

pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility

# nodes
nx.draw_networkx_nodes(G, pos, node_size=700)

# edges
nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
nx.draw_networkx_edges(
    G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
)

# node labels
nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
# edge weight labels
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels)

ax = plt.gca()
ax.margins(0.08)
plt.axis("off")
plt.tight_layout()
plt.show()