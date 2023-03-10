#%%
import networkx as nx
import numpy as np
from graph import Graph
import random
import matplotlib.pyplot as plt


#Create random graph and assign random weights (with a seed)
G = nx.erdos_renyi_graph(6, 0.5, seed = 5 )
random.seed(42)
for (u, v) in G.edges():
    G.edges[u,v]['weight'] = random.uniform(0.01,0.02)


sample_sizes = np.linspace(1,250, 4)


maxs = []

threshold = 5
k = 3
for i in sample_sizes:
    maximum = Graph().greedy_search(G, k, int(i), preset_s=0.3, repetitions = 10)
    maxs.append(maximum)
    

plt.plot(sample_sizes, maxs, 'x-', label = 'Uniform search',)
plt.xlabel('Samples')
plt.ylabel('Density')
plt.axhline(y = maxs[-1], color = 'grey', linestyle = '--', label = 'max density')
plt.legend()
#%%

maxs = []
for i in sample_sizes:
    maximum = Graph().uniform_search2(G,3, int(i))
    maxs.append(maximum)
#%%
plt.plot(sample_sizes, maxs, 'x-', label = 'Uniform search',)
plt.xlabel('Samples')
plt.ylabel('Density')
plt.axhline(y = maxs[-1], color = 'grey', linestyle = '--', label = 'max density')
plt.legend()


#%%
