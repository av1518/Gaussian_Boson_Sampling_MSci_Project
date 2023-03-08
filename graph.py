import networkx as nx
from scipy.sparse import csr_matrix
import numpy as np
from strawberryfields.decompositions import takagi
import random 
from tqdm import tqdm

class Graph():

    def adj_to_GBS(self, G_adj):
        '''Decomposes input adjacency using Takagi demposition in the form
        adj = U * diag(lambdas) * U^T 
        Returns: 
        lambdas, U'''
        lambdas,U = takagi(G_adj)
        return lambdas, U

    def nx_adj(self, G, node_list = None):
        '''Takes networkx graph G and returns adjacency matrix
        node_list = nodes to be considered (use to get adj of subgraph)'''
        if node_list == None:
            sparce = nx.adjacency_matrix(G)
            adj = sparce.toarray()
        else:
            sparce = nx.adjacency_matrix(G, node_list)
            adj = sparce.toarray()
        return adj

    def density(self, G):
        '''Takes input network G (networkx object) and returns density'''
        adj = self.nx_adj(G)
        V = len(adj)
        return 2 *  np.sum(adj) / ( abs(V) * (abs(V) - 1))

    def density_adj(self, adj):
        '''Takes an adjacency matrix adj and returns density'''
        V = len(adj)
        return 2 *  np.sum(adj) / ( abs(V) * (abs(V) - 1))

    def uniform_search(self, G, n, number_of_samples):
        '''n = subgraph size'''
        N = G.number_of_nodes()
        sample_d = []
        for i in tqdm(range(number_of_samples)):
            
            sample = random.sample(range(N), n)
            adj_of_sample = self.nx_adj(G, node_list = sample)
            d = self.density_adj(adj_of_sample)
            sample_d.append(d)
        return sample_d


    def uniform_search2(self, G, k, n, repetitions = 400):
        '''
        k = number of nodes in subgraph
        n = number of samples
        repetitions = number of repetitions for each sample size
        see kolt's paper for proceedure
        '''
        N = G.number_of_nodes()
        maxima = []
        for repetitions in tqdm(range(repetitions)):
            sample_d = []
            for i in range(n):
                sample = random.sample(range(N), k)
                adj_of_sample = self.nx_adj(G, node_list = sample)
                d = self.density_adj(adj_of_sample)
                sample_d.append(d)
            maxima.append(max(sample_d))
        avg_maximum = np.sum(maxima)/repetitions
        return avg_maximum







