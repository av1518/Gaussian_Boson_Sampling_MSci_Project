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

# %%

n_modes = 3

r_k = [0.5] * n_modes

U = unitary_group.rvs(n_modes) 

#%%
gbs = GBS_simulation()
greedy = Greedy()


ideal_marg_tor = Marginal().get_marginal_distribution_from_tor([0,1], U, r_k)

gredy_dist = greedy.get_S_matrix(n_modes, 100, 2, marginals)


cutoff = 6
loss = np.linspace(np.pi/2, np.pi, 10 )
for i in loss:  
    ideal_dist = gbs.get_noisy_marginals_from_simulation(n_modes, 6, r_k, U, i)


    



#%%
np.save('gbs_ideal_dist_modes_6', ideal_dist)
print(ideal_dist)
# %%
