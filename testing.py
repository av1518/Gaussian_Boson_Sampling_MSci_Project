#%%
from GBS_marginals import Marginal
from scipy.stats import unitary_group
import numpy as np


n_modes = 2


squeezing_params = np.random.uniform(0.2, 0.3, n_modes)
unitary = unitary_group.rvs(n_modes)

S = Marginal().get_S(squeezing_params)
# print(S)

sigma_in = Marginal().get_input_covariance_matrix(S)

# print(sigma_in)
#%%
# print(unitary)
# sigma = Marginal().get_output_covariance_matrix(unitary, squeezing_params)

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
