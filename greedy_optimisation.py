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

n_modes = 3
r_k = [0.4] * n_modes
U = unitary_group.rvs(n_modes) 
cutoff = 7

#%%
gbs = GBS_simulation()
greedy = Greedy()
probs = TheoreticalProbabilities()
n_samples = np.linspace(200,1000,10)

ideal_marg_tor = probs.get_all_ideal_marginals_from_torontonian(n_modes,r_k,U,2)


# distances = []
# for i in tqdm(loss):  
#     ideal_dist = gbs.get_noisy_marginal_from_simulation(n_modes, cutoff, r_k, U,list(range(n_modes)), i)
#     distance = total_variation_distance(ideal_dist, greedy_dist)
#     distances.append(distance)

distances = []
for i in tqdm(n_samples):  
    print(i)
    greedy_matrix = greedy.get_S_matrix(n_modes, int(i), 2, ideal_marg_tor)
    greedy_dist = greedy.get_distribution_from_outcomes(greedy_matrix)
    ideal_dist = gbs.get_noisy_marginal_from_simulation(n_modes, cutoff, r_k, U,list(range(n_modes)), 0)
    distance = total_variation_distance(ideal_dist, greedy_dist)
    distances.append(distance)

#%%

plt.figure()
plt.plot(n_samples, distances, 'o-', label = f'Modes = {n_modes}, Cutoff = {cutoff} ')
plt.xlabel('n_samples')
plt.ylabel(r'$\mathcal{D}$(Greedy,Ground)')
plt.legend()
plt.show()

np.save('distances_greedy,ground_n=3_cut=8', distances)
