#%%
import numpy as np
from utils import total_variation_distance
from gbs_simulation import GBS_simulation
from scipy.stats import unitary_group
from greedy import Greedy
from gbs_probabilities import TheoreticalProbabilities
from tqdm import tqdm
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit


plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.major.size'] = 50



n_modes = 5
r_k = [0.4] * n_modes
U = unitary_group.rvs(n_modes) 
cutoff = 7
loss = np.linspace(0, 1, 30 )
L = 1000

with plt.style.context(['science']):
    plt.figure(figsize=[9, 6])
    plt.xticks(size=16)
    plt.yticks(size=16)
    d6 =  np.load('distances_greedy,ground_n=5_cut=6_samples=1000_N=30.npy', allow_pickle=True)
    d7 = np.load('distances_greedy,ground_n=5_cut=7_samples=1000_N=30.npy', allow_pickle=True)
    plt.plot(loss, d6, 'o-', label=f'Modes = {n_modes}, Cutoff = 6, Samples = {L}', markersize=8)
    plt.plot(loss, d7, 'o-', label=f'Modes = {n_modes}, Cutoff = 7, Samples = {L}', markersize=8)
    plt.xlabel('Loss', fontsize=18)
    plt.ylabel(r'$\mathcal{D}$(Greedy,GBS)', fontsize=18)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('opt.png', dpi=600)
    plt.show()

# %% Optical loss with cutoff error
cutoff_error = abs(d7-d6)

with plt.style.context(['science']):
    plt.figure(figsize=[8,6])
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.tight_layout()
    # plt.title('Optical Loss Model', size = 25)
    


    d7 = np.load('distances_greedy,ground_n=5_cut=7_samples=1000_N=30.npy', allow_pickle=True)
    plt.errorbar(loss, d7, yerr=cutoff_error,label=f' Cutoff error',capsize = 7,color = 'black',linestyle = '' )
    plt.plot(loss, d7, 'o', label=f'Modes = {n_modes}, Cutoff = 7', markersize=7, color = 'black')
    plt.plot(loss, d7, '--', label=r'$\chi^2$ fit', markersize=7, color = 'firebrick', linewidth = 2)
    plt.xlabel('Optical loss fraction', fontsize=23)
    plt.ylabel(r'$\mathcal{D}$(Greedy,GBS)', fontsize=23)
    plt.legend(fontsize=20)

    plt.savefig('optical-loss-plot.png', dpi=600)
    plt.show()

#%% Gate model
cutoff = 6
range_n = 0.3
n_points = 30
stddev = np.linspace(0, range_n, n_points)
repetitions = 100
n_modes = 4

distances = np.load('distances_greedy,ground_n=4_cut=6,repetitions=100,range =0.3,_gate_error,numberofpoints= 30.npy',allow_pickle=True)

def f(x,a,b):
    return a*x**2 + b 

popt,pcov = curve_fit(f,stddev,distances)

with plt.style.context(['science']):
    plt.figure(figsize=[8, 6])
    plt.xticks(size=16)
    plt.yticks(size=16)


    plt.plot(stddev, distances, 'o-', label = f'Modes = {n_modes}, Cutoff = {cutoff}, Repetitions = {repetitions}', markersize=7, color = 'black'
    )
    plt.plot(stddev, f(stddev,popt[0],popt[1]),linestyle='--',color = 'firebrick',linewidth = 2, label = r'$\chi^2$ fit')
    plt.xlabel(r'Standard Deviation $\sigma$ ', fontsize = 20)
    plt.ylabel(r'$\overline{\mathcal{D}}$(Greedy,GBS)', fontsize = 20)


    plt.tight_layout()
    plt.legend(fontsize=19.5)
    plt.savefig('gate model plot.png', dpi=600)
    plt.show()

#%% Scaling plot
n_modes = 4
mean_n_photon = 0.1
cutoff = 7
loss = np.linspace(0, 0.5, 20)
L = 1500

distances = np.load('greedy_success_with_scaling_n=4_cut=7_mean_n_photon=0.1_L=1500.npy')

popt,pcov = curve_fit(f,loss,distances)

with plt.style.context(['science']):
  
    plt.figure(figsize=[8, 6])
    plt.title('Scaled Squeezing', fontsize=22)
    plt.xticks(size=16)
    plt.yticks(size=16)

    plt.plot(loss, distances, 'o-', label = f'Modes = {n_modes}, Cutoff = {cutoff} ', markersize=7, color = 'black')
    plt.plot(loss, f(loss,popt[0],popt[1]),linestyle='--',color = 'firebrick',linewidth = 2, label = r'$\chi^2$ fit')
    plt.xlabel('Loss',fontsize = 20)
    plt.ylabel(r'$\mathcal{D}$(Greedy,GBS)', fontsize = 20)
    plt.legend()


    plt.tight_layout()
    plt.legend(fontsize=20)
    plt.savefig('optical loss with scaling', dpi=400)
    plt.show()
#%% Distinguishability model
n_modes = 5
s = 0.5
r_k = [s] * n_modes
U = unitary_group.rvs(n_modes) 
cutoff = 7
s2 = np.linspace(0.0, 0.3, 30)
L = 2000

distances = np.load('greedy_success_distinguishability_n=5_cut=7_primary_squeezing=0.5_L=2000.npy')

d6 = np.load('greedy_success_distinguishability_n=5_cut=6_primary_squeezing=0.5_L=2000.npy', allow_pickle=True)
d7 = np.load('greedy_success_distinguishability_n=5_cut=7_primary_squeezing=0.5_L=2000.npy', allow_pickle=True)
error = 1/5 * abs(d7-d6)


with plt.style.context(['science']):
  
    plt.figure(figsize=[8, 6])
    # plt.title('Distinguishability model', fontsize=22)
    plt.xticks(size=16)
    plt.yticks(size=16)

    plt.plot(s2, d7, 'o', label = f'Modes = {n_modes}, Cutoff = {cutoff} ', markersize=7, color = 'black')
    plt.plot(s2, d7, '--', label = r'$\chi^2$ fit',color = 'firebrick', linewidth = 2.5)
    plt.errorbar(s2,d7,yerr= error, label=f' Cutoff error',capsize = 7,color = 'black',linestyle = '' )
    # plt.plot(loss, f(loss,popt[0],popt[1]),linestyle='--',color = 'firebrick',linewidth = 2, label = r'$\chi^2$ fit')
    plt.xlabel('Squeezing of imperfection',fontsize = 23)
    plt.ylabel(r'$\mathcal{D}$(Greedy,GBS)', fontsize = 23)
    plt.legend()


    plt.tight_layout()
    plt.legend(fontsize=20)
    plt.savefig('distinguishability-plot', dpi=500)
    plt.show()

#%%Adding error
n_modes = 5
s = 0.5
r_k = [s] * n_modes
U = unitary_group.rvs(n_modes) 
cutoff = 7
s2 = np.linspace(0.0, 0.3, 30)
L = 2000

distances = np.load('greedy_success_distinguishability_n=5_cut=7_primary_squeezing=0.5_L=2000.npy')

d6 = np.load('greedy_success_distinguishability_n=5_cut=6_primary_squeezing=0.5_L=2000.npy', allow_pickle=True)
d7 = np.load('greedy_success_distinguishability_n=5_cut=7_primary_squeezing=0.5_L=2000.npy', allow_pickle=True)
error = abs(d7-d6)


with plt.style.context(['science']):
  
    plt.figure(figsize=[8, 6])
    # plt.title('Distinguishability model', fontsize=22)
    plt.xticks(size=16)
    plt.yticks(size=16)

    plt.plot(s2, d7, 'o-', label = f'Modes = {n_modes}, Cutoff = 7', markersize=7, color = 'black')
    plt.plot(s2, d6, 'o-', label = f'Modes = {n_modes}, Cutoff = 6 ', markersize=7, color = 'black')
    # plt.errorbar(s2,d7,yerr= error, label=f' Cutoff error',capsize = 7,color = 'black',linestyle = '' )
    # plt.plot(loss, f(loss,popt[0],popt[1]),linestyle='--',color = 'firebrick',linewidth = 2, label = r'$\chi^2$ fit')
    plt.xlabel('Loss',fontsize = 20)
    plt.ylabel(r'$\mathcal{D}$(Greedy,GBS)', fontsize = 20)
    plt.legend()


    plt.tight_layout()
    plt.legend(fontsize=20)
    plt.savefig('distinguishability', dpi=400)
    plt.show()


# %%
def y4(mark):
    return (mark - (0.075*75.63 + 0.2*68.28 + 0.3625*69.82))/0.3625

