import numpy as np
import matplotlib.pyplot as plt
from greedy import Greedy
from scipy.stats import unitary_group
from gbs_simulation import GBS_simulation
from gbs_probabilities import TheoreticalProbabilities
from tqdm import tqdm

n_modes = 7
s = 0.5
squeezing_params = [s] * n_modes
unitary = unitary_group.rvs(n_modes, random_state=1)
L = 2000

probs = TheoreticalProbabilities()
k_greedy = list(range(1, 6))
k_order = list(range(1, n_modes + 1))

for k_order_fixed in tqdm(k_greedy):
    ideal_margs_fixed = probs.get_all_ideal_marginals_from_torontonian(n_modes, squeezing_params, unitary, k_order_fixed)
    S_matrix = Greedy().get_S_matrix(n_modes, L, k_order_fixed, ideal_margs_fixed)
    mean_dists = []
    for k in k_order:
        ideal_margs = probs.get_all_ideal_marginals_from_torontonian(n_modes, squeezing_params, unitary, k)
        greedy_marginal_dists = Greedy().get_marginal_distances_of_greedy_matrix(S_matrix, k, ideal_margs)
        marginal_dists = [x[1] for x in greedy_marginal_dists]
        mean_dist = np.sum(marginal_dists)/len(marginal_dists)
        mean_dists.append(mean_dist)
    plt.plot(k_order, mean_dists, label = f'Greedy order={k_order_fixed}')

plt.xlabel('Marginal order')
plt.ylabel('Mean Variation Distance')
plt.xticks(k_order)
plt.legend()
plt.show()

#%%
n_modes = 5
s = 0.5
squeezing_params = [s] * n_modes
unitary = unitary_group.rvs(n_modes, random_state=1)
L = 1200
cutoff = 7
s2 = 0.24

probs = TheoreticalProbabilities()
simul = GBS_simulation()
k_greedy = list(range(1, n_modes + 1))
k_order = list(range(1, n_modes + 1))

for k_order_fixed in tqdm(k_greedy):
    ideal_margs_fixed = probs.get_all_ideal_marginals_from_torontonian(n_modes, squeezing_params, unitary, k_order_fixed)
    S_matrix = Greedy().get_S_matrix(n_modes, L, k_order_fixed, ideal_margs_fixed)
    mean_dists = []
    for k in k_order:
        noisy_margs = simul.get_all_noisy_marginals_from_gaussian_simulation_with_distinguishability(n_modes, cutoff, squeezing_params, unitary, k, s2)
        greedy_marginal_dists = Greedy().get_marginal_distances_of_greedy_matrix(S_matrix, k, noisy_margs)
        marginal_dists = [x[1] for x in greedy_marginal_dists]
        mean_dist = np.sum(marginal_dists)/len(marginal_dists)
        mean_dists.append(mean_dist)
    plt.plot(k_order, mean_dists, label = f'Greedy order={k_order_fixed}')

plt.xlabel('Marginal order')
plt.ylabel('Mean Variation Distance')
plt.xticks(k_order)
plt.legend()
plt.show()
    