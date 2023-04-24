#%%
import numpy as np
from utils import total_variation_distance, kl_divergence
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
U = unitary_group.rvs(n_modes, random_state=1) 
cutoff = 8
s2 = np.linspace(0.0, 0.3, 30)
L = 2000

#%%
# gbs = GBS_simulation()
# greedy = Greedy()
# probs = TheoreticalProbabilities()

# ideal_marg_tor = probs.get_all_ideal_marginals_from_torontonian(n_modes,r_k,U,2)
# greedy_matrix = greedy.get_S_matrix(n_modes, L, 2, ideal_marg_tor)
# greedy_dist = greedy.get_distribution_from_outcomes(greedy_matrix)
# ideal_distr = gbs.get_ideal_marginal_from_gaussian_simulation(n_modes, cutoff, r_k, U,list(range(n_modes)))
# distance = total_variation_distance(ideal_distr, greedy_dist)
# print(distance)

# distances = []
# for i in tqdm(s2):  
#     ideal_dist = gbs.get_marginal_from_simulation_with_distinguishable_photons(n_modes, cutoff, r_k, U,list(range(n_modes)), i)
#     distance = total_variation_distance(ideal_dist, greedy_dist)
#     distances.append(distance)

# plt.figure()
# plt.plot(s2, distances, 'o-', label = f'Modes = {n_modes}, Cutoff = {cutoff} ')
# plt.xlabel('Squeezing of imperfection')
# plt.ylabel(r'$\mathcal{D}$(Greedy,Ground)')
# plt.legend()
# np.save(f'greedy_success_distinguishability_n={n_modes}_cut={cutoff}_primary_squeezing={s}_L={L}', distances)
# plt.show()
    
#%%
# gbs = GBS_simulation()
# greedy = Greedy()
# probs = TheoreticalProbabilities()

# ideal_marg_tor = probs.get_all_ideal_marginals_from_torontonian(n_modes,r_k,U,2)
# greedy_matrix = greedy.get_S_matrix(n_modes, L, 2, ideal_marg_tor)
# greedy_dist = greedy.get_distribution_from_outcomes(greedy_matrix)
# ideal_distr = gbs.get_ideal_marginal_from_gaussian_simulation(n_modes, cutoff, r_k, U,list(range(n_modes)))
# distance = kl_divergence(ideal_distr, greedy_dist)
# print(distance)

# distances = []
# for i in tqdm(s2):  
#     ideal_dist = gbs.get_marginal_from_simulation_with_distinguishable_photons(n_modes, cutoff, r_k, U,list(range(n_modes)), i)
#     distance = kl_divergence(ideal_dist, greedy_dist)
#     distances.append(distance)

# plt.figure()
# plt.plot(s2, distances, 'o-', label = f'Modes = {n_modes}, Cutoff = {cutoff} ')
# plt.xlabel('Squeezing of imperfection')
# plt.ylabel(r'$\mathcal{D}$(Greedy,Ground)')
# plt.legend()
# np.save(f'greedy_kl_success_distinguishability_n={n_modes}_cut={cutoff}_primary_squeezing={s}_L={L}', distances)
# plt.show()

# d7 = np.load('greedy_success_distinguishability_n=5_cut=7_primary_squeezing=0.5_L=2000.npy', allow_pickle=True)
# d8 = np.load('greedy_success_distinguishability_n=5_cut=8_primary_squeezing=0.5_L=2000.npy', allow_pickle=True)
# error = 1/5 * abs(d8-d7)


# with plt.style.context(['science']):
#     plt.figure(figsize=[8, 6])
#     plt.xticks(size=16)
#     plt.yticks(size=16)
#     plt.plot(s2, d8, 'o', label = f'Modes = {n_modes}, Cutoff = {cutoff} ', markersize=7, color = 'black')
#     plt.plot(s2, d8, '--', label = r'$\chi^2$ fit',color = 'firebrick', linewidth = 2.5)
#     plt.errorbar(s2,d8,yerr= error, label=f' Cutoff error',capsize = 7,color = 'black',linestyle = '' )
#     plt.xlabel('Squeezing of imperfection',fontsize = 20)
#     plt.ylabel(r'$\mathcal{D}$(Greedy,Ground)', fontsize = 20)
#     plt.legend()
#     plt.tight_layout()
#     plt.legend(fontsize=18)
#     plt.savefig('distinguishability_dist_plot', dpi=600)
#     plt.show()

#%%

d7 = np.load('greedy_kl_success_distinguishability_n=5_cut=7_primary_squeezing=0.5_L=2000.npy', allow_pickle=True)
d8 = np.load('greedy_kl_success_distinguishability_n=5_cut=8_primary_squeezing=0.5_L=2000.npy', allow_pickle=True)
error = 1/5 * abs(d8-d7)


with plt.style.context(['science']):
    plt.figure(figsize=[8, 6])
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.plot(s2, d8, 'o', label = f'Modes = {n_modes}, Cutoff = {cutoff} ', markersize=7, color = 'black')
    plt.plot(s2, d8, '--', label = r'$\chi^2$ fit',color = 'firebrick', linewidth = 2.5)
    plt.errorbar(s2,d8,yerr= error, label=f' Cutoff error',capsize = 7,color = 'black',linestyle = '' )
    plt.xlabel('Squeezing of imperfection',fontsize = 20)
    plt.ylabel(r'$\mathcal{KL}$(Greedy,Ground)', fontsize = 20)
    plt.legend()
    plt.tight_layout()
    plt.legend(fontsize=18)
    plt.savefig('distinguishability_kl_plot', dpi=600)
    plt.show()