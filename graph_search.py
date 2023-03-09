#%%
import networkx as nx
import numpy as np
from graph import Graph
import random
import matplotlib.pyplot as plt

#%%
#Create random graph and assign random weights (with a seed)
G = nx.erdos_renyi_graph(10, 0.8, seed = 5 )
random.seed(42)
for (u, v) in G.edges():
    G.edges[u,v]['weight'] = random.random()


sample_sizes = np.linspace(1,250, 50)

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


