#%%
import numpy as np
from utils import total_variation_distance
from gbs_simulation import GBS_simulation
from scipy.stats import unitary_group
from greedy import Greedy
from gbs_probabilities import TheoreticalProbabilities
from tqdm import tqdm
import matplotlib.pyplot as plt 

# %%

n_modes = 4
r_k = [0.5] * n_modes
U = unitary_group.rvs(n_modes) 

#%%
gbs = GBS_simulation()
greedy = Greedy()
probs = TheoreticalProbabilities()

ideal_marg_tor = probs.get_all_ideal_marginals_from_torontonian(n_modes,r_k,U,2)
greedy_matrix = greedy.get_S_matrix(n_modes, 500, 2, ideal_marg_tor)
greedy_dist = greedy.get_distribution_from_outcomes(greedy_matrix)

cutoff = 5
loss = np.linspace(0, 1, 15 )

distances = []
for i in tqdm(loss):  
    ideal_dist = gbs.get_noisy_marginal_from_simulation(n_modes, cutoff, r_k, U,list(range(n_modes)), i)
    distance = total_variation_distance(ideal_dist, greedy_dist)
    distances.append(distance)

plt.plot(loss, distances, 'o-', label = f'modes = {n_modes}, cutoff = {cutoff} ')
plt.xlabel(r'Noise ($\phi$) ')
plt.ylabel(r'$\mathcal{D}$(Greedy,Ground)')
plt.legend()
# plt.xlim(-0.3,0.8)
# plt.ylim(0,0.10)


#%%
np.save('distances_greedy,ground_n=3_cut=8', distances)

#%% Gate error model plot
gbs = GBS_simulation()
greedy = Greedy()
probs = TheoreticalProbabilities()

ideal_marg_tor = probs.get_all_ideal_marginals_from_torontonian(n_modes,r_k,U,2)
greedy_matrix = greedy.get_S_matrix(n_modes, 500, 2, ideal_marg_tor)
greedy_dist = greedy.get_distribution_from_outcomes(greedy_matrix)

cutoff = 6
stddev = np.linspace(0, 5, 100  )

distances = []
for i in tqdm(stddev):  
    ideal_dist = gbs.get_noisy_marginal_gate_error(cutoff, r_k, U, list(range(n_modes)), i)
    distance = total_variation_distance(ideal_dist, greedy_dist)
    distances.append(distance)


plt.plot(stddev, distances, 'o-', label = f'modes = {n_modes}, cutoff = {cutoff} ')
plt.xlabel(r'Standard Deviation ')
plt.ylabel(r'$\mathcal{D}$(Greedy,Ground)')
plt.legend()

# plt.xlim(-0.3,0.8)
# plt.ylim(0,0.10)


#%%
np.save(f'distances_greedy,ground_n={n_modes}_cut={cutoff}_gate_error', distances)
plt.savefig('gate error model vs greedy')