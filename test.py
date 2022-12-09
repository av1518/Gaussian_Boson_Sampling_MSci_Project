from qutip.states import fock_dm, coherent_dm
from qutip.wigner import wigner
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

x = np.linspace(-5,5,200)
dim = 5

vac_dm = fock_dm(dim, 0)
W_vacuum = wigner(vac_dm, x, x)

fig, axes = plt.subplots(1, 1, figsize=(9,9))
cont0 = axes.contourf(x, x, W_vacuum, 100)
plt.show()

def get_prob_all_zero_bitstring(output_dm, xvec):
    vac_dm = fock_dm(output_dm.shape[0], 0)
    integrand = np.dot(wigner(output_dm, xvec, xvec), wigner(vac_dm, xvec, xvec))
    prob_zero = integrate.simpson(integrand, xvec)
    return prob_zero

dm = coherent_dm(10, np.sqrt(2))
print(get_prob_all_zero_bitstring(dm, x))