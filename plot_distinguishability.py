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

n_modes = 5
s = 0.5
r_k = [s] * n_modes
U = unitary_group.rvs(n_modes) 
cutoff = 7
s2 = np.linspace(0.0, 0.3, 30)
L = 2000

#%%
gbs = GBS_simulation()
greedy = Greedy()
probs = TheoreticalProbabilities()

ideal_marg_tor = probs.get_all_ideal_marginals_from_torontonian(n_modes,r_k,U,2)
greedy_matrix = greedy.get_S_matrix(n_modes, L, 2, ideal_marg_tor)
greedy_dist = greedy.get_distribution_from_outcomes(greedy_matrix)
ideal_distr = gbs.get_ideal_marginal_from_gaussian_simulation(n_modes, cutoff, r_k, U,list(range(n_modes)))
distance = total_variation_distance(ideal_distr, greedy_dist)
print(distance)

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
np.save(f'greedy_success_distinguishability_n={n_modes}_cut={cutoff}_primary_squeezing={s}_L={L}', distances)
plt.show()