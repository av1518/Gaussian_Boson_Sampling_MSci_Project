import numpy as np
import matplotlib.pyplot as plt
from greedy import Greedy
from scipy.stats import unitary_group
from gbs_simulation import GBS_simulation
from gbs_probabilities import TheoreticalProbabilities
from tqdm import tqdm

n_modes = 7
s = 0.5
squeezing_params = [s] * n_modes
unitary = unitary_group.rvs(n_modes, random_state=1)
L = 2000

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
#         greedy_marginal_dists = Greedy().get_marginal_kl_divergences_of_greedy_matrix(S_matrix, k, ideal_margs)
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

# np.save(f'high_order_correlations_kl_n={n_modes}_squeezing={s}_L={L}_up_to_{k_greedy[-1]}th_order_approx', full_mean_dists)


plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.major.width"] = 1.5
plt.rcParams["ytick.major.width"] = 1.5
plt.rcParams["xtick.major.size"] = 50

full_mean_dists = np.load(
    "high_order_correlations_kl_n=7_squeezing=0.5_L=2000_up_to_5th_order_approx.npy"
)
k_greedy = list(range(1, 6))
k_order = list(range(1, n_modes + 1))

with plt.style.context(["science"]):
    plt.figure(figsize=[8, 6])
    plt.xticks(k_order, size=20)
    plt.yticks(size=20)
    plt.tight_layout()
    for i in range(len(k_greedy)):
        plt.plot(
            k_order,
            full_mean_dists[i],
            "o-",
            label=f"Approximation order = {k_greedy[i]}",
            linewidth=2.5,
            markersize=7,
        )
    plt.xlabel("Marginal Order", fontsize=24)
    plt.ylabel(r"$\mathcal{\overline{KL}}$(Greedy,GBS)", fontsize=24)
    plt.legend(fontsize=20)
    plt.savefig("high_order_correlations_kl_plot.png", dpi=600)
    plt.show()
