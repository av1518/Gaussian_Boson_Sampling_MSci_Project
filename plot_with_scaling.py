#%%
import numpy as np
import matplotlib.pyplot as plt
from greedy import Greedy
from scipy.stats import unitary_group
from gbs_simulation import GBS_simulation
import math
from scipy.optimize import fsolve
from greedy import Greedy
from tqdm import tqdm
import matplotlib.pyplot as plt 
from utils import total_variation_distance, kl_divergence

gbs = GBS_simulation()
greedy = Greedy()


#%% Single mode mean photon number

def mean_photon_number_single_mode(r, n):
    return (n*math.factorial(2*n)/(2*np.cosh(r)*(math.factorial(n))**2))*(((np.tanh(r))**2)/4)**n

def mean_photon_number_single_mode_simplified(r, n):
    return (1/(np.sqrt(4*np.pi)*np.cosh(r)))*np.sqrt(n)*((np.tanh(r))**2)**n

squeezing = 0.6
n= np.arange(1, 6, 1)
n_bar = [mean_photon_number_single_mode(squeezing, i) for i in n]
plt.figure()
plt.plot(n, n_bar, label=f'r={squeezing}')
plt.xlabel('n')
plt.ylabel('Single-mode Mean Photon Number')
plt.legend()

n_bar = [mean_photon_number_single_mode_simplified(squeezing, i) for i in n]
plt.figure()
plt.plot(n, n_bar, label=f'r={squeezing}')
plt.xlabel('n')
plt.ylabel('Single-mode Mean Photon Number')
plt.legend()
#plt.show()

#%% Can always just consider the first 6 terms in the sum of the single-mode mean photon
# number, as the higher order terms have negligible contributions.

def total_mean_photon_number(loss, n_modes, r):
    """Assume constant loss in every mode."""
    loss_angle = loss*np.pi/2
    return n_modes*((np.cos(loss_angle))**2)*np.sum(
        [(1/(np.sqrt(4*np.pi)*np.cosh(r)))*np.sqrt(n)*((np.tanh(r))**2)**n for n in np.arange(1, 6, 1)])

def get_scaled_squeezing(tot_mean_n_photon, n_modes, loss):
    def func(squeezing):
        return tot_mean_n_photon - total_mean_photon_number(loss, n_modes, squeezing)
    initial_guess = 0.6
    root = fsolve(func, initial_guess)
    if np.isclose(func(root[0]), 0.0):
        return root[0]

n_modes = 5
mean_n_photon = 0.15
U = unitary_group.rvs(n_modes, random_state=1) 
cutoff = 7
loss = np.linspace(0, 0.5, 20)
ideal_squeezing = [get_scaled_squeezing(mean_n_photon, n_modes, 0)]*n_modes
L = 1500

s = [get_scaled_squeezing(mean_n_photon, n_modes, i) for i in loss]
print(s)

#%%
distances = []
for i in tqdm(loss):
    squeezing = [get_scaled_squeezing(mean_n_photon, n_modes, i)]*n_modes
    marginals = gbs.get_all_lossy_marginals_from_gaussian_simulation(n_modes, cutoff, squeezing, U, 2, i)
    greedy_matrix = greedy.get_S_matrix(n_modes, L, 2, marginals)
    greedy_distr = greedy.get_distribution_from_outcomes(greedy_matrix)
    print('Total mean photon number:', total_mean_photon_number(i, n_modes, squeezing[0]))
    ground_distr = gbs.get_lossy_marginal_from_gaussian_simulation(n_modes, cutoff, squeezing, U,list(range(n_modes)), i)
    distance = total_variation_distance(ground_distr, greedy_distr)
    distances.append(distance)
    print('----')

plt.figure()
plt.plot(loss, distances, 'o-', label = f'Modes = {n_modes}, Cutoff = {cutoff} ')
plt.xlabel('Loss')
plt.ylabel('Distance(Greedy,Ground)')
plt.legend()
np.save(f'greedy_success_with_scaling_n={n_modes}_cut={cutoff}_mean_n_photon={mean_n_photon}_L={L}', distances)
plt.show()

