import numpy as np
from scipy.stats import unitary_group
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.major.size'] = 50

#%%
# n_modes = 4
# mean_n_photon = 0.1
# U = unitary_group.rvs(n_modes, random_state=1) 
# cutoff = 7
# loss = np.linspace(0, 0.5, 20)
# L = 1500

# d6 = np.load('greedy_success_with_scaling_n=4_cut=6_mean_n_photon=0.1_L=1500.npy', allow_pickle=True)
# d7 = np.load('greedy_success_with_scaling_n=4_cut=7_mean_n_photon=0.1_L=1500.npy', allow_pickle=True)
# error = 1/5 * abs(d7-d6)

# def f(x,a,b):
#     return a*np.exp((-b*x**2)/2)

# popt,pcov = curve_fit(f,loss,d7)

# with plt.style.context(['science']):
#     plt.figure(figsize=[8, 6])
#     plt.xticks(size=16)
#     plt.yticks(size=16)
#     plt.plot(loss, d7, 'o', label = f'Modes = {n_modes}, Cutoff = {cutoff} ', markersize=7, color = 'black')
#     plt.plot(loss, f(loss,popt[0],popt[1]), '--', label = r'$\chi^2$ fit',color = 'firebrick', linewidth = 2.5)
#     plt.errorbar(loss,d7,yerr= error, label=f' Cutoff error',capsize = 7,color = 'black',linestyle = '' )
#     plt.xlabel('Loss Fraction',fontsize = 20)
#     plt.ylabel(r'$\mathcal{D}$(Greedy,Ground)', fontsize = 20)
#     plt.legend()
#     plt.tight_layout()
#     plt.legend(fontsize=18)
#     plt.savefig('results_distance_optical_loss_with_scaling_plot', dpi=600)
#     plt.show()

#%%

n_modes = 4
s = 0.5
r_k = [s] * n_modes
U = unitary_group.rvs(n_modes, random_state=1) 
cutoff = 8
loss = np.linspace(0, 1, 30)
n_fixed = 1
L = 2000

d7 = np.load('greedy_success_fixed_n_clicks_distance_n=4_cut=7_n_clicks=1_L=2000_s=0.5.npy', allow_pickle=True)
d8 = np.load('greedy_success_fixed_n_clicks_distance_n=4_cut=8_n_clicks=1_L=2000_s=0.5.npy', allow_pickle=True)
k7 = np.load('greedy_success_fixed_n_clicks_kl_n=4_cut=7_n_clicks=1_L=2000_s=0.5.npy', allow_pickle=True)
k8 = np.load('greedy_success_fixed_n_clicks_kl_n=4_cut=8_n_clicks=1_L=2000_s=0.5.npy', allow_pickle=True)

error = 1/5 * abs(d8-d7)

with plt.style.context(['science']):
    plt.figure(figsize=[8, 6])
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.plot(loss, d8, 'o', label = f'Modes = {n_modes}, Cutoff = {cutoff} ', markersize=7, color = 'black')
    plt.plot(loss, d8, '--', label = r'$\chi^2$ fit',color = 'firebrick', linewidth = 2.5)
    plt.errorbar(loss,d8,yerr= error, label=f' Cutoff error',capsize = 7,color = 'black',linestyle = '' )
    plt.xlabel('Loss Fraction',fontsize = 20)
    plt.ylabel(r'$\mathcal{D}$(Greedy,Ground)', fontsize = 20)
    plt.legend()
    plt.tight_layout()
    plt.legend(fontsize=18)
    plt.savefig('results_distance_optical_loss_fixed_n_clicks_plot', dpi=600)

    plt.figure(figsize=[8, 6])
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.plot(loss, k8, 'o', label = f'Modes = {n_modes}, Cutoff = {cutoff} ', markersize=7, color = 'black')
    plt.plot(loss, k8, '--', label = r'$\chi^2$ fit',color = 'firebrick', linewidth = 2.5)
    plt.errorbar(loss,k8,yerr= error, label=f' Cutoff error',capsize = 7,color = 'black',linestyle = '' )
    plt.xlabel('Loss Fraction',fontsize = 20)
    plt.ylabel(r'$\mathcal{KL}$(Greedy,Ground)', fontsize = 20)
    plt.legend()
    plt.tight_layout()
    plt.legend(fontsize=18)
    plt.savefig('results_kl_optical_loss_fixed_n_clicks_plot', dpi=600)
    plt.show()
