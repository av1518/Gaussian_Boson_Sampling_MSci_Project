import numpy as np
import matplotlib.pyplot as plt
from greedy import Greedy
from scipy.stats import unitary_group
from gbs_simulation import GBS_simulation
from gbs_probabilities import TheoreticalProbabilities
from tqdm import tqdm

# n_modes = 7
# s = 0.5
# squeezing_params = [s] * n_modes
# unitary = unitary_group.rvs(n_modes, random_state=1)
# L = 2000

# probs = TheoreticalProbabilities()
# k_greedy = list(range(1, 6))
# k_order = list(range(1, n_modes + 1))

# full_mean_dists = []
# for k_order_fixed in tqdm(k_greedy):
#     ideal_margs_fixed = probs.get_all_ideal_marginals_from_torontonian(n_modes, squeezing_params, unitary, k_order_fixed)
#     S_matrix = Greedy().get_S_matrix(n_modes, L, k_order_fixed, ideal_margs_fixed)
#     mean_dists = []
#     for k in k_order:
#         ideal_margs = probs.get_all_ideal_marginals_from_torontonian(n_modes, squeezing_params, unitary, k)
#         greedy_marginal_dists = Greedy().get_marginal_distances_of_greedy_matrix(S_matrix, k, ideal_margs)
#         marginal_dists = [x[1] for x in greedy_marginal_dists]
#         mean_dist = np.sum(marginal_dists)/len(marginal_dists)
#         mean_dists.append(mean_dist)
#     full_mean_dists.append(mean_dists)
#     plt.plot(k_order, mean_dists, label = f'Greedy order={k_order_fixed}')

# plt.xlabel('Marginal order')
# plt.ylabel('Mean Variation Distance')
# plt.xticks(k_order)
# plt.legend()
# plt.show()

# np.save(f'high_order_correlations_n={n_modes}_squeezing={s}_L={L}_up_to_{k_greedy[-1]}th_order_approx', full_mean_dists)

#%%
n_modes = 5
s = 0.5
squeezing_params = [s] * n_modes
unitary = unitary_group.rvs(n_modes, random_state=1)
L = 2000
cutoff = 8
s2 = 0.25

# probs = TheoreticalProbabilities()
# simul = GBS_simulation()
# k_greedy = list(range(2, 4))
# k_order = list(range(1, n_modes + 1))
# full_mean_dists_ideal = []
# full_mean_dists_noisy = []
# full_mean_divs_ideal = []
# full_mean_divs_noisy = []

# for k_order_fixed in tqdm(k_greedy):
#     ideal_margs_fixed = probs.get_all_ideal_marginals_from_torontonian(n_modes, squeezing_params, unitary, k_order_fixed)
#     S_matrix = Greedy().get_S_matrix(n_modes, L, k_order_fixed, ideal_margs_fixed)
#     mean_dists_ideal = []
#     mean_dists_noisy = []
#     mean_divs_ideal = []
#     mean_divs_noisy = []
#     for k in k_order:
#         ideal_margs = probs.get_all_ideal_marginals_from_torontonian(n_modes, squeezing_params, unitary, k)
#         greedy_marginal_dists = Greedy().get_marginal_distances_of_greedy_matrix(S_matrix, k, ideal_margs)
#         greedy_marginal_divs = Greedy().get_marginal_kl_divergences_of_greedy_matrix(S_matrix, k, ideal_margs)
#         marginal_dists = [x[1] for x in greedy_marginal_dists]
#         marginal_divs = [x[1] for x in greedy_marginal_divs]
#         mean_dist = np.sum(marginal_dists)/len(marginal_dists)
#         mean_div = np.sum(marginal_divs)/len(marginal_divs)
#         mean_dists_ideal.append(mean_dist)
#         mean_divs_ideal.append(mean_div)

#         noisy_margs = simul.get_all_noisy_marginals_from_gaussian_simulation_with_distinguishability(n_modes, cutoff, squeezing_params, unitary, k, s2)
#         greedy_marginal_dists2 = Greedy().get_marginal_distances_of_greedy_matrix(S_matrix, k, noisy_margs)
#         greedy_marginal_divs2 = Greedy().get_marginal_kl_divergences_of_greedy_matrix(S_matrix, k, noisy_margs)
#         marginal_dists2 = [x[1] for x in greedy_marginal_dists2]
#         marginal_divs2 = [x[1] for x in greedy_marginal_divs2]
#         mean_dist2 = np.sum(marginal_dists2)/len(marginal_dists2)
#         mean_div2 = np.sum(marginal_divs2)/len(marginal_divs2)
#         mean_dists_noisy.append(mean_dist2)
#         mean_divs_noisy.append(mean_div2)
    
#     full_mean_dists_ideal.append(mean_dists_ideal)
#     full_mean_dists_noisy.append(mean_dists_noisy)
#     full_mean_divs_ideal.append(mean_divs_ideal)
#     full_mean_divs_noisy.append(mean_divs_noisy)
#     plt.plot(k_order, mean_dists_ideal, label = f'Greedy order (ideal)={k_order_fixed}')
#     plt.plot(k_order, mean_dists_noisy, label = f'Greedy order (noisy)={k_order_fixed}')

# plt.xlabel('Marginal order')
# plt.ylabel('Mean Variation Distance')
# plt.xticks(k_order)
# plt.legend()
# plt.show()

# np.save(f'high_order_correlations_ideal_n={n_modes}_squeezing={s}_s2={s2}_L={L}_approx_order={k_greedy[0]},{k_greedy[1]}', full_mean_dists_ideal)
# np.save(f'high_order_correlations_noisy_n={n_modes}_squeezing={s}_s2={s2}_L={L}_approx_order={k_greedy[0]},{k_greedy[1]}', full_mean_dists_noisy)
# np.save(f'high_order_correlations_kl_ideal_n={n_modes}_squeezing={s}_s2={s2}_L={L}_approx_order={k_greedy[0]},{k_greedy[1]}', full_mean_divs_ideal)
# np.save(f'high_order_correlations_kl_noisy_n={n_modes}_squeezing={s}_s2={s2}_L={L}_approx_order={k_greedy[0]},{k_greedy[1]}', full_mean_divs_noisy)


#%%

# n_modes = 5
# s = 0.5
# squeezing_params = [s] * n_modes
# unitary = unitary_group.rvs(n_modes, random_state=1)
# L = 1200
# cutoff = 7
# s2 = 0.24

# probs = TheoreticalProbabilities()
# simul = GBS_simulation()
# k_greedy = list(range(1, n_modes + 1))
# k_order = list(range(1, n_modes + 1))

# for k_order_fixed in tqdm(k_greedy):
#     ideal_margs_fixed = probs.get_all_ideal_marginals_from_torontonian(n_modes, squeezing_params, unitary, k_order_fixed)
#     S_matrix = Greedy().get_S_matrix(n_modes, L, k_order_fixed, ideal_margs_fixed)
#     mean_dists = []
#     for k in k_order:
#         noisy_margs = simul.get_all_noisy_marginals_from_gaussian_simulation_with_distinguishability(n_modes, cutoff, squeezing_params, unitary, k, s2)
#         greedy_marginal_dists = Greedy().get_marginal_distances_of_greedy_matrix(S_matrix, k, noisy_margs)
#         marginal_dists = [x[1] for x in greedy_marginal_dists]
#         mean_dist = np.sum(marginal_dists)/len(marginal_dists)
#         mean_dists.append(mean_dist)
#     plt.plot(k_order, mean_dists, label = f'Greedy order={k_order_fixed}')
    
# plt.xlabel('Marginal order')
# plt.ylabel('Mean Variation Distance')
# plt.xticks(k_order)
# plt.legend()
# plt.show()

# plt.rcParams['axes.facecolor']='white'
# plt.rcParams['savefig.facecolor']='white'
# plt.rcParams['axes.linewidth'] = 1.5
# plt.rcParams['xtick.major.width'] = 1.5
# plt.rcParams['ytick.major.width'] = 1.5
# plt.rcParams['xtick.major.size'] = 50

# full_mean_dists = np.load('high_order_correlations_n=7_squeezing=0.5_L=2000_up_to_5th_order_approx.npy')
# k_greedy = list(range(1, 6))
# k_order = list(range(1, n_modes + 1))

# with plt.style.context(['science']):
#     plt.figure(figsize=[8,6])
#     plt.xticks(k_order, size=16)
#     plt.yticks(size=16)
#     plt.tight_layout()
#     for i in range(len(k_greedy)):
#         plt.plot(k_order, full_mean_dists[i], 'o-', label = f'Approximation order = {k_greedy[i]}', linewidth=2.5, markersize=7)
#     plt.xlabel('Marginal Order', fontsize=20)
#     plt.ylabel('Mean Total Variation Distance', fontsize=20)
#     plt.legend(fontsize=16)
#     plt.savefig('high_order_correlations_plot.png', dpi=600)
#     plt.show()

#%%
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.major.size'] = 50

full_mean_dists_ideal = np.load('high_order_correlations_ideal_n=5_squeezing=0.5_s2=0.25_L=2000_approx_order=2,3.npy')
full_mean_dists_noisy = np.load('high_order_correlations_noisy_n=5_squeezing=0.5_s2=0.25_L=2000_approx_order=2,3.npy')
full_mean_divs_ideal = np.load('high_order_correlations_kl_ideal_n=5_squeezing=0.5_s2=0.25_L=2000_approx_order=2,3.npy')
full_mean_divs_noisy = np.load('high_order_correlations_kl_noisy_n=5_squeezing=0.5_s2=0.25_L=2000_approx_order=2,3.npy')

k_greedy = list(range(2, 4))
k_order = list(range(1, n_modes + 1))

with plt.style.context(['science']):
    plt.figure(figsize=[8,6])
    plt.xticks(k_order, size=16)
    plt.yticks(size=16)
    plt.tight_layout()
    for i in range(len(k_greedy)):
        plt.plot(k_order, full_mean_dists_ideal[i], 'o-', label = f'Order {k_greedy[i]} approximation, ideal', linewidth=2.5, markersize=7)
        plt.plot(k_order, full_mean_dists_noisy[i], 'o-', label = f'Order {k_greedy[i]} approximation, noisy', linewidth=2.5, markersize=7)
    plt.xlabel('Marginal Order', fontsize=20)
    plt.ylabel('Mean Total Variation Distance', fontsize=20)
    plt.legend(fontsize=16)
    plt.savefig('high_order_correlations_comparison_ideal-noisy_plot.png', dpi=600)

    plt.figure(figsize=[8,6])
    plt.xticks(k_order, size=16)
    plt.yticks(size=16)
    plt.tight_layout()
    for i in range(len(k_greedy)):
        plt.plot(k_order, full_mean_divs_ideal[i], 'o-', label = f'Order {k_greedy[i]} approximation, ideal', linewidth=2.5, markersize=7)
        plt.plot(k_order, full_mean_divs_noisy[i], 'o-', label = f'Order {k_greedy[i]} approximation, noisy', linewidth=2.5, markersize=7)
    plt.xlabel('Marginal Order', fontsize=20)
    plt.ylabel('Mean KL Divergence', fontsize=20)
    plt.legend(fontsize=16)
    plt.savefig('high_order_correlations_kl_comparison_ideal-noisy_plot.png', dpi=600)
    plt.show()
    