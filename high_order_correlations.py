import numpy as np
import matplotlib.pyplot as plt
from greedy import Greedy
from scipy.stats import unitary_group
from gbs_simulation import GBS_simulation

k_order_fixed = 2
n_modes = 4
cutoff = 5
s = 0.4
squeezing_params = [s] * n_modes
unitary = unitary_group.rvs(n_modes, random_state=1)
L = 1000

simul = GBS_simulation()
ideal_margs_fixed = simul.get_all_ideal_marginals_from_gaussian_simulation(n_modes, cutoff, squeezing_params, unitary, k_order_fixed)
S_matrix = Greedy().get_S_matrix(n_modes, L, k_order_fixed, ideal_margs_fixed)

k_order = list(range(2, n_modes + 1))
mean_dists = []
for k in k_order:
    ideal_margs = simul.get_all_ideal_marginals_from_gaussian_simulation(n_modes, cutoff, squeezing_params, unitary, k)
    greedy_marginal_dists = Greedy().get_marginal_distances_of_greedy_matrix(S_matrix, k, ideal_margs)
    marginal_dists = [x[1] for x in greedy_marginal_dists]
    mean_dist = np.sum(marginal_dists)/len(marginal_dists)
    mean_dists.append(mean_dist)

plt.figure()
plt.plot(k_order, mean_dists)
plt.xlabel('Marginal order')
plt.ylabel('Mean Variation Distance')
plt.xticks(k_order)
plt.show()
    