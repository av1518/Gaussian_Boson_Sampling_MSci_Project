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
loss = np.linspace(0, 1, 10 )

#%%
gbs = GBS_simulation()
greedy = Greedy()
probs = TheoreticalProbabilities()

ideal_marg_tor = probs.get_all_ideal_marginals_from_torontonian(n_modes,r_k,U,2)
greedy_matrix = greedy.get_S_matrix(n_modes, 500, 2, ideal_marg_tor)
greedy_dist = greedy.get_distribution_from_outcomes(greedy_matrix)

distances = []
for i in tqdm(loss):  
    ideal_dist = gbs.get_noisy_marginal_from_simulation(n_modes, cutoff, r_k, U,list(range(n_modes)), i)
    distance = total_variation_distance(ideal_dist, greedy_dist)
    distances.append(distance)

plt.figure()
plt.plot(loss, distances, 'o-', label = f'Modes = {n_modes}, Cutoff = {cutoff} ')
plt.xlabel('Loss')
plt.ylabel(r'$\mathcal{D}$(Greedy,Ground)')
plt.legend()
# plt.show()
# plt.xlim(-0.3,0.8)
# plt.ylim(0,0.10)

# np.save('distances_greedy,ground_n=3_cut=8', distances)

# %% Scale squeezing parameters to account for photon loss

def total_mean_photon_number(loss, squeezing_params):
    """Assume constant loss in every mode."""
    loss_angle = loss*np.pi/2
    number = [((np.cos(loss_angle))**2)*(np.sinh(s))**2 for s in squeezing_params]
    return np.sum(number)

def scaled_squeezing(mean_photon_number, n_modes, loss):
    """Assume constant squeezing and constant loss in every mode."""
    loss_angle = loss*np.pi/2
    return np.arcsinh(np.sqrt(mean_photon_number/(n_modes*(np.cos(loss_angle))**2)))

n_modes = 3
mean_n_photon = 1.0
U = unitary_group.rvs(n_modes) 
cutoff = 7
loss = np.linspace(0, 0.5, 8)
ideal_squeezing = [scaled_squeezing(mean_n_photon, n_modes, 0)]*n_modes

s = [scaled_squeezing(mean_n_photon, n_modes, i) for i in loss]
print(s)

#%%
gbs = GBS_simulation()
greedy = Greedy()
probs = TheoreticalProbabilities()

ideal_marg_tor = probs.get_all_ideal_marginals_from_torontonian(n_modes,ideal_squeezing,U,2)
greedy_matrix = greedy.get_S_matrix(n_modes, 500, 2, ideal_marg_tor)
greedy_dist = greedy.get_distribution_from_outcomes(greedy_matrix)

distances = []
all_zero_probs_with_scaling = []
all_zero_probs_no_scaling = []

for i in tqdm(loss):
    squeezing = [scaled_squeezing(mean_n_photon, n_modes, i)]*n_modes
    print('Total mean photon number:', total_mean_photon_number(i, squeezing))
    ideal_dist = gbs.get_noisy_marginal_from_simulation(n_modes, cutoff, squeezing, U,list(range(n_modes)), i)
    ideal_dist_no_scaling = gbs.get_noisy_marginal_from_simulation(n_modes, cutoff, ideal_squeezing, U,list(range(n_modes)), i)
    all_zero_probs_with_scaling.append(ideal_dist[0]) 
    all_zero_probs_no_scaling.append(ideal_dist_no_scaling[0])
    distance = total_variation_distance(ideal_dist, greedy_dist)
    distances.append(distance)

plt.figure()
plt.plot(loss, distances, 'o-', label = f'Modes = {n_modes}, Cutoff = {cutoff} ')
plt.xlabel('Loss')
plt.ylabel(r'$\mathcal{D}$(Greedy,Ground)')
plt.legend()

plt.figure()
plt.plot(loss, all_zero_probs_no_scaling, 'o-', label = 'Ground without scaling')
plt.plot(loss, all_zero_probs_with_scaling, 'o-', label = 'Ground with scaling')
plt.plot(loss, [greedy_dist[0]]*len(loss), label = 'Greedy')
plt.xlabel('Loss')
plt.ylabel('Probability of all-zero bitstring')
plt.legend()
plt.show()