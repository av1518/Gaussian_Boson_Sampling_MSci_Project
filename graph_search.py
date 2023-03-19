#%%
import networkx as nx
import numpy as np
from graph import Graph
import random
import matplotlib.pyplot as plt


#Create random graph and assign random weights (with a seed)
G = nx.erdos_renyi_graph(4, 0.5, seed = 5 ) 
random.seed(42)
for (u, v) in G.edges():
    G.edges[u,v]['weight'] = random.uniform(0.005,0.010)


sample_sizes = np.linspace(1, 25, 20)
int_ns = [int(x) for x in sample_sizes]


cut = 10
k = 2

maxima = Graph().greedy_search2(G, k, n_range= int_ns, repetitions = 10)
#%%
np.save(f'maxima_from_greedy_search_range', maxima)
#%%
plt.plot(sample_sizes, maxima, 'x-', label = 'Uniform search',)
plt.xlabel('Samples')
plt.ylabel('Density')
plt.axhline(y = maxima[-1], color = 'grey', linestyle = '--', label = 'max density')
plt.legend()


#%%
maxs.append(maximum)
    

plt.plot(sample_sizes, maxs, 'x-', label = 'Uniform search',)
plt.xlabel('Samples')
plt.ylabel('Density')
plt.axhline(y = maxs[-1], color = 'grey', linestyle = '--', label = 'max density')
plt.legend()


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


sample_sizes = np.linspace(1,100, 4)
int_ns = [int(x) for x in sample_sizes]


maxs = []

cut = 9
k = 3
for i in sample_sizes:
    maximum = Graph().greedy_search2(G, k, n_range= int_ns, cutoff=cut, repetitions = 10)
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
plt.plot(sample_sizes, maxima, 'x-', label = 'Greedy',)
plt.xlabel('Samples')
plt.ylabel('Density')
# plt.axhline(y = maxs[-1], color = 'grey', linestyle = '--', label = 'max density')
plt.legend()


#%%
