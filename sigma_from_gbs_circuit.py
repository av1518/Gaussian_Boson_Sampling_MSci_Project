#%%
from scipy.stats import unitary_group
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *
import thewalrus

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


# %%
