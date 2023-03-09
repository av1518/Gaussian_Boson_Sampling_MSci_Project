#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gbs_probabilities import TheoreticalProbabilities
from greedy import Greedy
from scipy.stats import unitary_group
from gbs_simulation import GBS_simulation
from tqdm import tqdm
import thewalrus
import scipy as sp

#%% Test greedy algorithm

marginals = np.array(
    [[[0,1], [0.25, 0.25, 0.25, 0.25]],
     [[0,2], [0.25, 0.25, 0.25, 0.25]],
     [[1,2], [0.25, 0.25, 0.25, 0.25]],
     [[0,3], [0.25, 0.25, 0.25, 0.25]],
     [[1,3], [0.25, 0.25, 0.25, 0.25]],
     [[2,3], [0.25, 0.25, 0.25, 0.25]]])
n_modes = 4
k_order = 2

# print(Greedy().get_S_matrix(n_modes, 20, k_order, marginals))

#%% Test theoretical calculation of marginal distributions

r_k = [
1.6518433645720738,
1.687136454610338,
1.62938385974034,
1.706029877650956,
1.8395638626723685,
1.3943570412472834,
1.4819924169286014,
1.6313980669381827,
1.6559961541267325,
1.3389267197532349,
1.568736620327057,
1.6772334549978614,
1.459031307907052,
1.4124223294979523,
1.3440269631323098,
1.4328684458997072,
1.4675334685180914,
1.6270874674912998,
1.6044404863902908,
1.581538415101846,
1.6519035066626184,
1.5456532234514821,
1.5974577318822245,
1.7043797524114164,
1.7294783286655087]

#r_k = np.array(r_k + r_k)
# r_k = np.random.uniform(0.05, 0.15, 25)
# T_re = pd.read_excel('matrix_re.xlsx', header = None).to_numpy()
# T_im = pd.read_excel('matrix_im.xlsx', header = None).to_numpy()
# T = T_re + T_im * 1j
# T = T.T


#%% Replicate GBS using marginals obtained from simulation

k_order = 2
n_modes = 3
cutoff = 4
squeezing_params = np.random.uniform(0.2, 0.3, n_modes)
unitary = unitary_group.rvs(n_modes)

simul = GBS_simulation()
ideal_margs = simul.get_all_ideal_marginals_from_gaussian_simulation(n_modes, cutoff, squeezing_params, unitary, k_order)
S_matrix = Greedy().get_S_matrix(n_modes, 1000, k_order, ideal_margs)
greedy_marginal_dists = Greedy().get_marginal_distances_of_greedy_matrix(S_matrix, k_order, ideal_margs)
print(greedy_marginal_dists)

ideal_full_distr = np.array(simul.get_ideal_marginal_from_gaussian_simulation(n_modes, cutoff, squeezing_params, unitary, list(range(n_modes))))
noisy_full_distr = np.array(simul.get_lossy_marginal_from_gaussian_simulation(n_modes, cutoff, squeezing_params, unitary, list(range(n_modes))))
greedy_full_dist = Greedy().get_distribution_from_outcomes(S_matrix)
ideal_total_dist = 0.5*np.sum(np.abs(ideal_full_distr - greedy_full_dist))
noisy_total_dist = 0.5*np.sum(np.abs(noisy_full_distr - greedy_full_dist))
print('Distance between ideal and greedy:', ideal_total_dist)
print('Distance between noisy and greedy:', noisy_total_dist)

#%% Test ideal marginals against torontonian:

n_modes = 2
squeezing_params = np.random.uniform(0.2, 0.3, n_modes)
unitary = unitary_group.rvs(n_modes)
ideal_marg_tor = TheoreticalProbabilities().get_marginal_distribution_from_tor([0,1], unitary, squeezing_params)
ideal_marg_simul = simul.get_ideal_marginal_from_gaussian_simulation(n_modes, 15, squeezing_params, unitary, [0,1])
print('Marginal from torontonian:', ideal_marg_tor)
print('Marginal from simulation:', ideal_marg_simul)

#%% Test lossy and ideal marginals in different backends:

n_modes = 3
U = unitary_group.rvs(n_modes, random_state=1) 
cutoff = 6
loss = 0.0
s = 0.5

squeezing = [s]*n_modes
print('Ideal:', simul.get_ideal_marginal_from_gaussian_simulation(n_modes, cutoff, squeezing, U,list(range(n_modes))))
print('Lossy Fock:', simul.get_lossy_marginal_from_fock_simulation(n_modes, cutoff, squeezing, U,list(range(n_modes)), loss))
# Marginals obtained when using the LossChannel are not accuarte.
print('Lossy Gaussian:', simul.get_lossy_marginal_from_gaussian_simulation(n_modes, cutoff, squeezing, U,list(range(n_modes)), loss))

print('Ideal marginals:', simul.get_all_ideal_marginals_from_gaussian_simulation(n_modes, cutoff, squeezing, U, 2))
print('Lossy marginals:', simul.get_all_lossy_marginals_from_gaussian_simulation(n_modes, cutoff, squeezing, U, 2, loss))

