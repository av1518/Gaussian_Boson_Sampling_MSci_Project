import networkx as nx
from scipy.sparse import csr_matrix
import numpy as np
from strawberryfields.decompositions import takagi
import random 
from tqdm import tqdm
from gbs_simulation import GBS_simulation
from gbs_probabilities import TheoreticalProbabilities
from greedy import Greedy
import copy
from scipy.optimize import fsolve



class Graph():
    def __init__(self):
        self.extra_samples= []
        self.sl = [] #sample list

    def get_submatrix_with_fixed_n_clicks(
        self,
        S_matrix: np.ndarray,
        n: int
    ) -> np.ndarray:
        submatrix = []
        for elem in S_matrix:
            count = np.count_nonzero(elem == 1.0)
            if count == n:
                submatrix.append(elem)
        return np.array(submatrix)


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
        if node_list is None:
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
        std = np.std(maxima)
        return avg_maximum,std

    def find_ones(self, lst):
        indices = []
        for i, val in enumerate(lst):
            if val == 1:
                indices.append(i)
        return indices

    def get_reduced_adj(self, sigma: np.ndarray, sample: list) -> np.ndarray:
        '''
        Parameters
        ----------
        sigma: covariance matrix
        R: mode indices considered

        Returns the reduced matrix associated with the input mode indices.
        '''
        indices = self.find_ones(sample)
        sigma_red = np.empty((len(indices),len(indices)))
        for r_index, row in enumerate(indices):
            for c_index, column in enumerate(indices):
                sigma_red[r_index, c_index] = sigma[row, column]
        return sigma_red


    def total_mean_photon_number(self, loss, n_modes, r):
        """Assume constant loss in every mode."""
        loss_angle = loss*np.pi/2
        return n_modes*((np.cos(loss_angle))**2)*np.sum(
            [(1/(np.sqrt(4*np.pi)*np.cosh(r)))*np.sqrt(n)*((np.tanh(r))**2)**n for n in np.arange(1, 6, 1)])



    def get_scaled_squeezing(self, tot_mean_n_photon, n_modes, loss):
        def func(squeezing):
            return tot_mean_n_photon - self.total_mean_photon_number(loss, n_modes, squeezing)
        initial_guess = 0.6
        root = fsolve(func, initial_guess)
        if np.isclose(func(root[0]), 0.0):
            return root[0] #this is scaled squezzing




    def greedy_search(self, G, k, n, repetitions = 400, L = 2000):

        adj = self.nx_adj(G)
        N = G.number_of_nodes()
        s_i, U = self.adj_to_GBS(adj)
        print(f's_i = {s_i}')

        # s_tuned = [preset_s] * N
        s_ideal = [self.get_scaled_squeezing(k/N + 0.2, N, 0)] * N
        print(f's_ideal= {s_ideal}')


        probs = TheoreticalProbabilities()
        ideal_margs = probs.get_all_ideal_marginals_from_torontonian(N, s_ideal, U, 2)
        print('here1')
        maxima = []

        for repetitions in tqdm(range(repetitions)):
            print(f'length of extra samples = {len(self.extra_samples)}')
            if len(self.extra_samples) >= n:
                greedy_samples = self.extra_samples[:n]
                del self.extra_samples[:n]
                print('got extra samples and deleted')
            else:
                S_matrix = Greedy().get_S_matrix(N, L, 2, ideal_margs)
                print(f'S matrix generated with L = {L} ')
                # print(S_matrix)
                # print('k=',k)
                greedy_samples_array = self.get_submatrix_with_fixed_n_clicks(S_matrix, k)
                greedy_samples = list(greedy_samples_array)
                self.extra_samples += greedy_samples
                # print(f'now here, greedy samples = {greedy_samples}')
            


            while (len(self.extra_samples)) < n:
                print(f'more samples for n={n}')
                more_S_matrix = Greedy().get_S_matrix(N, 2000, 2, ideal_margs)
                more_greedy_samples = list(self.get_submatrix_with_fixed_n_clicks(more_S_matrix, k))
                # print('more greedy samples before concatenation =', more_greedy_samples)
                # print('dimension of more samples=', more_greedy_samples.shape)
                greedy_samples += more_greedy_samples
                self.extra_samples += more_greedy_samples
                # np.concatenate((greedy_samples,more_greedy_samples))
                
                
            if len(greedy_samples) >= n:
                # print('the samples = ', greedy_samples)
                greedy_samples_n = greedy_samples[:n]
                self.extra_samples += greedy_samples[n:] 

            
            sample_d = []
            for i in range(n):
                sample = greedy_samples_n[i]
                adj_of_sample = self.get_reduced_adj(adj, sample)
                d = self.density_adj(adj_of_sample)
                sample_d.append(d)
            maxima.append(max(sample_d))
        avg_maximum = np.sum(maxima)/repetitions
        return avg_maximum

    def greedy_search2(self, G, k, n_range, repetitions = 400, L = 2000):
        adj = self.nx_adj(G)
        N = G.number_of_nodes()
        s_i, U = self.adj_to_GBS(adj)
        print(f's_i = {s_i}')
        # s_ideal = [self.get_scaled_squeezing(k/N + 0.3, N, 0)] * N
        s_ideal = [self.get_scaled_squeezing(k, N, 0)] * N
        print(f's_ideal= {s_ideal}')
        probs = TheoreticalProbabilities()
        ideal_margs = probs.get_all_ideal_marginals_from_torontonian(N, s_ideal, U, 2)
        print('here1')
        avg_maxima = []
        std_devs = []

        for n in n_range:
            print(f'now at n={n}')
            maxima_for_this_n = []
            
            for repetition in tqdm(range(repetitions)):
                
                print(f'length of extra samples = {len(self.sl)}')

                while len(self.sl) < n:
                    S_matrix = Greedy().get_S_matrix(N, L, 2, ideal_margs)
                    print(f'S matrix generated with L = {L} ')
                    subset = list(self.get_submatrix_with_fixed_n_clicks(S_matrix, k))
                    print(f'length of subset = {len(subset)}')
                    self.sl += subset

                samples_for_this_n = self.sl[:n]
                del self.sl[:n]
                print('got samples from sl and deleted')

                sample_d = []
                for i in range(n):
                    sample = samples_for_this_n[i]
                    # print(sample)
                    d = self.density_adj(self.get_reduced_adj(adj, sample))
                    # print(d)
                    sample_d.append(d)
                maxima_for_this_n.append(max(sample_d))
            avg_maximum_for_this_n = np.sum(maxima_for_this_n)/repetitions
            std_for_this_n = np.std(maxima_for_this_n)

            avg_maxima.append(avg_maximum_for_this_n)
            std_devs.append(std_for_this_n)
        return avg_maxima, std_devs
            
            
        










