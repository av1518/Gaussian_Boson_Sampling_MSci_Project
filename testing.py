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
