import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from GBS_marginals import Marginal
from greedy import Greedy
from scipy.stats import unitary_group
from ideal_gbs_simulation import GBS_simulation

#%% Test greedy algorithm

marginals = np.array(
    [[0.2, 0.25, 0.3, 0.25],
    [0.35, 0.2, 0.25, 0.2],
    [0.2, 0.3, 0.4, 0.1]])
n_modes = 4
k_order = 2

dists = []
for j in range(20, 180, 40):
    test_matrix = Greedy().get_S_matrix(n_modes, j, k_order, marginals)
    dist_j = Greedy().get_k_mode_marginal_distances_of_greedy_matrix(test_matrix, k_order, marginals)
    dists.append(dist_j)

marginal_index = np.arange(0, len(dists[0]), 1)

with plt.style.context(['science']):
    for i in range(len(dists)):
        plt.plot(marginal_index, dists[i], label=f'L = {20 + 40*i}')

    plt.xlabel('Marginal index')
    plt.ylabel('Variation distance')
    plt.xticks(marginal_index)
    plt.legend()
    #plt.show()
    #plt.savefig('variation_dist.png', dpi=400)

#%% Test theoretical calculation of marginal distributions

r_k = [
1.6518433645720738,
1.687136454610338,
1.62938385974034,
1.706029877650956,
1.8395638626723685,
1.3943570412472834,
1.4819924169286014,
1.6313980669381827,
1.6559961541267325,
1.3389267197532349,
1.568736620327057,
1.6772334549978614,
1.459031307907052,
1.4124223294979523,
1.3440269631323098,
1.4328684458997072,
1.4675334685180914,
1.6270874674912998,
1.6044404863902908,
1.581538415101846,
1.6519035066626184,
1.5456532234514821,
1.5974577318822245,
1.7043797524114164,
1.7294783286655087]

#r_k = np.array(r_k + r_k)
r_k = np.random.uniform(0.05, 0.15, 25)
T_re = pd.read_excel('matrix_re.xlsx', header = None).to_numpy()
T_im = pd.read_excel('matrix_im.xlsx', header = None).to_numpy()
T = T_re + T_im * 1j
T = T.T
U = unitary_group.rvs(50)

ideal_marg = Marginal().get_marginal_distribution([1,7], U, r_k)
noisy_marg = Marginal().get_marginal_distribution([1,7], T, r_k)

print('Ideal marginal:', ideal_marg)
print('Sum of ideal marginal:', sum(ideal_marg))
print('Noisy marginal:', noisy_marg)
print('Sum of noisy marginal:', sum(noisy_marg))

#%% Test calculation of marginals from simulation

n_modes = 3
cutoff = 10
squeezing_params = np.random.uniform(0.4, 0.6, n_modes)
unitary = unitary_group.rvs(n_modes)

simul = GBS_simulation()
print('Marginal from simulation:', simul.get_ideal_marginal(n_modes, cutoff, squeezing_params, unitary, [0,1]))