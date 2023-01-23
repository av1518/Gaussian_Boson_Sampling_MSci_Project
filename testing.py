#%%
from GBS_marginals import Marginal
from scipy.stats import unitary_group
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *
import thewalrus

n_modes = 2


r_k = np.random.uniform(0.2, 0.3, n_modes)
unitary = unitary_group.rvs(n_modes)

S = Marginal().get_S(r_k)
# print(S)

sigma_in = Marginal().get_input_covariance_matrix(S)


#%% Trying to get covariance matrix from Guassian backend


gbs = sf.Program(2)

with gbs.context as q:
    Sgate(r_k[0]) | q[0]
    Sgate(r_k[1]) | q[1]
    Interferometer(unitary) | q

eng = sf.Engine(backend = "gaussian")

state = eng.run(gbs).state
sigma = state.cov()
print(sigma)

print(thewalrus.quantum.is_valid_cov(sigma))
#%%

p = sf.Program(len(r_k))
n_modes = 3
r_k = np.random.uniform(0.2, 0.3, n_modes)
unitary = unitary_group.rvs(n_modes)


with p.context as q:
    for i, r in enumerate(r_k):
        Sgate(r) | q[i]
    Interferometer(unitary) | q

e = sf.Engine(backend = "gaussian")

state = e.run(p).state
sigma = state.cov()
print(sigma)

print(thewalrus.quantum.is_valid_cov(sigma))




#%%Function method does not work for some reason


def apply_squeezing_gates(r_k, unitary):
    gbss = sf.Program(2)

    with gbss.context() as q:
        for i, r in enumerate(r_k):
            Sgate(r) | q[i]
        Interferometer(unitary) | q


    eng = sf.Engine(backend = "gaussian")
    state = eng.run(gbss).state
    sigma = state.cov()
    print(sigma)
    print(thewalrus.quantum.is_valid_cov(sigma))
    
#%%
r_k = np.random.uniform(0.2, 0.3, n_modes)
unitary = unitary_group.rvs(n_modes)

apply_squeezing_gates(r_k,unitary)



#%% Function that builds a circuit based on input r_k and U:
def sigma_from_gbs(r_k, U):
    n = len(r_k)
    gbs = sf.Program(n)

    with gbs.context as q:




# print(sigma_in)
#%%
# print(unitary)
# sigma = Marginal().get_output_covariance_matrix(unitary, r_k)

# %%
U = unitary_group.rvs(5)
reduced = Marginal().get_reduced_matrix(U,[0,1])
# print(U)
# print(reduced)

#%%
U = unitary_group.rvs(4)
Bs = Marginal().get_reduced_B(U,[0,2])
print('u =',U)
print('bs=',Bs)
