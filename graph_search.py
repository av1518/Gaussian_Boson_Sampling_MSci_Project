#%%
import networkx as nx
import numpy as np
from graph import Graph
import random
import matplotlib.pyplot as plt


#%%


G = nx.Graph()

elist = [(1, 2, 5.0), (2, 3, 3.0), (1, 3, 1.0), (2, 4, 7.3)]
G.add_weighted_edges_from(elist)
adj = nx.adjacency_matrix(G)


#%%
print(adj)

G_adj = adj.toarray()
print(G_adj)


def density(G):
    '''Takes input network G (networkx object) and returns density'''
    sparce = nx.adjacency_matrix(G)
    adj = sparce.toarray()
    V = len(adj)
    return 2 *  np.sum(adj) / ( abs(V) * (abs(V) - 1))

density(G)

lambdas, U = takagi(G_adj)


# %%
print(nx.density(G))
print(density(G))

subgraph_adj = nx.adjacency_matrix(G, [])
sub_adj = subgraph_adj.toarray()
print(sub_adj)

#%%



#%%
uniform_samples = Graph().uniform_search(G,3,100)
print(uniform_samples)
#%%
cumu = np.cumsum(uniform_samples)

averages = []
for i in range(len(cumu)):
    if i == 0:
        av_0 = cumu[0]
        averages.append(av_0)
    else:
        av_i = cumu[i]/i
        averages.append(av_i)

#%%
plt.plot(range(len(uniform_samples)), averages)

#%%

maximum = Graph().uniform_search2(G,3, 100)
print(maximum)
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


