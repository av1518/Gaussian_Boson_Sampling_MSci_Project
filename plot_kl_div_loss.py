#%%
import numpy as np
from utils import kl_divergence
from gbs_simulation import GBS_simulation
from scipy.stats import unitary_group
from greedy import Greedy
from gbs_probabilities import TheoreticalProbabilities
from tqdm import tqdm
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit


gbs = GBS_simulation()
greedy = Greedy()
probs = TheoreticalProbabilities()

# %%

n_modes = 5
r_k = [0.4] * n_modes
U = unitary_group.rvs(n_modes) 
cutoff = 7
n_points = 30
loss = np.linspace(0, 1, n_points )
L = 2000
#%%

ideal_marg_tor = probs.get_all_ideal_marginals_from_torontonian(n_modes,r_k,U,2)
greedy_matrix = greedy.get_S_matrix(n_modes, L, 2, ideal_marg_tor)
greedy_dist = greedy.get_distribution_from_outcomes(greedy_matrix)

distances = []
for i in tqdm(loss):  
    ideal_dist = gbs.get_lossy_marginal_from_gaussian_simulation(n_modes, cutoff, r_k, U,list(range(n_modes)), i)
    distance = kl_divergence(ideal_dist, greedy_dist)
    distances.append(distance)

#%%
# distances = np.load('distances_greedy,ground_n=5_cut=7_samples=1000.npy',allow_pickle=True)

plt.figure()
plt.plot(loss, distances, 'o-', label = f'Modes = {n_modes}, Cutoff = {cutoff}, Samples = {L} ')
plt.xlabel('Loss')
plt.ylabel(r'$\mathcal{D}$(Greedy,Ground)')
plt.legend()
plt.show()

np.save(f'distances_kl_greedy,ground_n={n_modes}_cut={cutoff}_samples={L}_N={n_points}', distances)
