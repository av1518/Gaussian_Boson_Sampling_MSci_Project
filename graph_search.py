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
G = nx.erdos_renyi_graph(nodes, 0.5, seed = 5 ) 
random.seed(42)
for (u, v) in G.edges():
    G.edges[u,v]['weight'] = random.uniform(0.01,0.02)


sample_sizes = np.linspace(1, 30, 20)
int_ns = [int(x) for x in sample_sizes]


cut = 10
k = 3

maxima, errors = Graph().greedy_search2(G, k, n_range= int_ns, repetitions = 40)
#%%
code = 5
np.save(f'greedy_search_modes,code={code}', maxima)
#%%
plt.plot(sample_sizes, maxima, 'x-', label = 'greedy search',)
plt.xlabel('Samples')
plt.ylabel('Density')
plt.axhline(y = maxima[-1], color = 'grey', linestyle = '--', label = 'max density')
plt.legend()


#%%

maxs = []
for i in sample_sizes:
    maximum = Graph().uniform_search2(G, k, int(i))
    maxs.append(maximum)
#%%


plt.title(f'k={k},nodes = 6')
plt.plot(sample_sizes, maxs, '.-', label = 'Uniform search',)
plt.errorbar(sample_sizes, maxima, yerr = errors, label=f' Greedy search error', capsize = 3,color = 'purple',linestyle = '')
plt.plot(sample_sizes, maxima, '.-', label = 'Greedy', color = 'purple')
plt.axhline(y = maxima[-1], color = 'grey', linestyle = '--', label = 'max density')
plt.xlabel('Samples')
plt.ylabel('Density')
# plt.axhline(y = maxs[-1], color = 'grey', linestyle = '--', label = 'max density')
plt.legend()


#%%
with plt.style.context(['science']):
  
    plt.figure(figsize=[8, 6])
    # plt.title('Distinguishability model', fontsize=22)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.title(f'k={k},nodes = 6')
    plt.plot(sample_sizes, maxs, '.-', label = 'Uniform search',)
    plt.errorbar(sample_sizes, maxima, yerr = errors, label=f' Greedy search error', capsize = 3,color = 'purple',linestyle = '')
    plt.plot(sample_sizes, maxima, '.-', label = 'Greedy', color = 'purple')
    plt.axhline(y = maxima[-1], color = 'grey', linestyle = '--', label = 'max density')
    plt.xlabel('Samples',fontsize = 20)
    plt.ylabel('Density',fontsize = 20)
    # plt.axhline(y = maxs[-1], color = 'grey', linestyle = '--', label = 'max density')
    plt.legend(fontsize=20)
    plt.savefig(f'greedy_graph_search_code={code}', dpi=600)