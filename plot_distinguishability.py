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
cutoff = 7
s2 = np.linspace(0, 0.7, 10)

#%%
gbs = GBS_simulation()
greedy = Greedy()
probs = TheoreticalProbabilities()

ideal_marg_tor = probs.get_all_ideal_marginals_from_torontonian(n_modes,r_k,U,2)
greedy_matrix = greedy.get_S_matrix(n_modes, 500, 2, ideal_marg_tor)
greedy_dist = greedy.get_distribution_from_outcomes(greedy_matrix)

distances = []
for i in tqdm(s2):  
    ideal_dist = gbs.get_marginal_from_simulation_with_distinguishable_photons(n_modes, cutoff, r_k, U,list(range(n_modes)), i)
    distance = total_variation_distance(ideal_dist, greedy_dist)
    distances.append(distance)

plt.figure()
plt.plot(s2, distances, 'o-', label = f'Modes = {n_modes}, Cutoff = {cutoff} ')
plt.xlabel('Squeezing of imperfection')
plt.ylabel(r'$\mathcal{D}$(Greedy,Ground)')
plt.legend()
plt.show()