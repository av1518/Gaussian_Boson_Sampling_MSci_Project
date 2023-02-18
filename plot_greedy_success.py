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
plt.show()

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
mean_n_photon = 0.8
U = unitary_group.rvs(n_modes) 
cutoff = 7
loss = np.linspace(0, 0.6, 10)
ideal_squeezing = [scaled_squeezing(mean_n_photon, n_modes, 0)]*n_modes

s = [scaled_squeezing(mean_n_photon, n_modes, i) for i in loss]
print(s)

#%%
gbs = GBS_simulation()
greedy = Greedy()
probs = TheoreticalProbabilities()

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
distances = []

for i in tqdm(loss):
    squeezing = [scaled_squeezing(mean_n_photon, n_modes, i)]*n_modes
    marginals = gbs.get_all_noisy_marginals_from_simulation(n_modes, cutoff, squeezing, U, 2, i)
    greedy_matrix = greedy.get_S_matrix(n_modes, 700, 2, marginals)
    greedy_distr = greedy.get_distribution_from_outcomes(greedy_matrix)
    print('Total mean photon number:', total_mean_photon_number(i, squeezing))
    ground_distr = gbs.get_noisy_marginal_from_simulation(n_modes, cutoff, squeezing, U,list(range(n_modes)), i)
    distance = total_variation_distance(ground_distr, greedy_distr)
    distances.append(distance)

plt.figure()
plt.plot(loss, distances, 'o-', label = f'Modes = {n_modes}, Cutoff = {cutoff} ')
plt.xlabel('Loss')
plt.ylabel(r'$\mathcal{D}$(Greedy,Ground)')
plt.legend()
plt.show()
