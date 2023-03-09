#%%
from collections import Counter
import numpy as np
from utils import total_variation_distance
from gbs_simulation import GBS_simulation
from scipy.stats import unitary_group
from greedy import Greedy
from gbs_probabilities import TheoreticalProbabilities
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import bitstring_to_int

n_modes = 5
r_k = [0.5] * n_modes
U = unitary_group.rvs(n_modes, random_state=1) 
cutoff = 7
loss = np.linspace(0, 1, 10 )
n_fixed = int(n_modes/2)
L = 2000
#%%
gbs = GBS_simulation()
greedy = Greedy()

probs = TheoreticalProbabilities()

def get_submatrix_with_fixed_n_clicks(
    S_matrix: np.ndarray,
    n: int
) -> np.ndarray:
    submatrix = []
    for elem in S_matrix:
        count = np.count_nonzero(elem == 1.0)
        if count == n:
            submatrix.append(elem)
    return np.array(submatrix)

def get_distribution_from_outcomes(samples: np.ndarray) -> np.ndarray:
    """Turns list of outcomes (bitstrings) into empirical distribution."""
    bitstrings = [tuple(x) for x in samples]
    sorted_decimal_list = np.sort([bitstring_to_int(binary) for binary in bitstrings])
    count_dict = Counter(sorted_decimal_list)
    counts = [count_dict.get(i, 0) for i in range(2**len(samples[0]))]
    counts = [x for x in counts if x!= 0]
    distribution = np.array(counts) / np.sum(counts)
    return list(count_dict.keys()), distribution

ideal_marg_tor = probs.get_all_ideal_marginals_from_torontonian(n_modes,r_k,U,2)
greedy_matrix = greedy.get_S_matrix(n_modes, L, 2, ideal_marg_tor)
fixed_n_clicks_submatrix = get_submatrix_with_fixed_n_clicks(greedy_matrix, n_fixed)
subset, greedy_distr = get_distribution_from_outcomes(fixed_n_clicks_submatrix)
print('Bitstring subset (in decimal):', subset)

#%%

distances = []
for i in tqdm(loss):  
    ground_distr = gbs.get_lossy_marginal_from_gaussian_simulation(n_modes, cutoff, r_k, U,list(range(n_modes)), i)
    conditional_probs = [x for i, x in enumerate(ground_distr) if i in subset]
    ground_distr = np.array(conditional_probs)/np.sum(conditional_probs)
    distance = total_variation_distance(ground_distr, greedy_distr)
    distances.append(distance)

plt.figure()
plt.plot(loss, distances, 'o-', label = f'Modes = {n_modes}, Cutoff = {cutoff} ')
plt.xlabel('Loss')
plt.ylabel(r'$\mathcal{D}$(Greedy,Ground)')
plt.legend()
plt.show()

#%%
n_modes = 5
r_k = [0.5] * n_modes
U = unitary_group.rvs(n_modes) 
cutoff = 7
n_fixed = np.arange(0, n_modes + 1, 1)

ideal_marg_tor = probs.get_all_ideal_marginals_from_torontonian(n_modes,r_k,U,2)
greedy_matrix = greedy.get_S_matrix(n_modes, 1000, 2, ideal_marg_tor)

distances = []
for n in tqdm(n_fixed):
    fixed_n_clicks_submatrix = get_submatrix_with_fixed_n_clicks(greedy_matrix, n)
    subset, greedy_distr = get_distribution_from_outcomes(fixed_n_clicks_submatrix)
    ideal_distr = gbs.get_ideal_marginal_from_gaussian_simulation(n_modes, cutoff, r_k, U, list(range(n_modes)))
    conditional_probs = [x for i, x in enumerate(ideal_distr) if i in subset]
    ideal_distr = np.array(conditional_probs)/np.sum(conditional_probs)
    distance = total_variation_distance(ideal_distr, greedy_distr)
    distances.append(distance)

plt.figure()
plt.plot(n_fixed, distances, 'o-', label = f'Modes = {n_modes}, Cutoff = {cutoff} ')
plt.xlabel('N clicks')
plt.ylabel(r'$\mathcal{D}$(Greedy,Ground)')
plt.xticks(n_fixed)
plt.legend()
plt.show()