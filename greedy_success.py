#%%
import strawberryfields as sf
from strawberryfields import ops
import numpy as np
from typing import Tuple, List
import itertools as iter
from utils import bitstring_to_int, convert_to_clicks
from itertools import combinations
from gbs_simulation import GBS_simulation
from scipy.stats import unitary_group
from greedy import Greedy
from GBS_marginals import Marginal
from tqdm import tqdm
import matplotlib.pyplot as plt 

# %%

n_modes = 4

r_k = [0.5] * n_modes

U = unitary_group.rvs(n_modes) 

#%%
gbs = GBS_simulation()
greedy = Greedy()


ideal_marg_tor = Marginal().get_ideal_marginals_from_torontonian(n_modes,r_k,U,2)


greedy_matrix = greedy.get_S_matrix(n_modes, 500, 2, ideal_marg_tor)
greedy_dist = greedy.get_distribution_from_outcomes(greedy_matrix)

cutoff = 5
loss = np.linspace(0, np.pi/2, 15 )

distances = []
for i in tqdm(loss):  
    ideal_dist = gbs.get_noisy_marginal_from_simul(n_modes, cutoff, r_k, U,list(range(n_modes)), i)
    distance = 0.5 * np.sum(np.abs(ideal_dist - greedy_dist))
    distances.append(distance)




plt.plot(loss, distances, 'o-', label = f'modes = {n_modes}, cutoff = {cutoff} ')
plt.xlabel(r'Noise ($\phi$) ')
plt.ylabel(r'$\mathcal{D}$(Greedy,Ground)')
plt.legend()
# plt.xlim(-0.3,0.8)
# plt.ylim(0,0.10)


#%%
np.save('distances_greedy,ground_n=3_cut=8', distances)

# %%
