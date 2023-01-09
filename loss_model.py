import strawberryfields as sf
from strawberryfields import ops
from scipy.stats import unitary_group
import numpy as np

n_modes = 2
cutoff = 7
squeezing_params = np.random.uniform(0.4, 0.6, n_modes)
unitary = unitary_group.rvs(n_modes)

prog = sf.Program(2*n_modes)
with prog.context as q:
    for i, s in enumerate(squeezing_params):
        ops.Sgate(s) | q[i]
    for cmd in ops.Interferometer(unitary).decompose(tuple([qubit for qubit in q[:n_modes]])):
        cmd.op | cmd.reg
    for i, qubit in enumerate(q[:n_modes]):
        ops.BSgate() | (qubit, q[n_modes + i]) 

U = sf.utils.extract_unitary(prog, cutoff_dim=cutoff, vectorize_modes=True)
#print(U.shape)
print(np.dot(U, U.conj().T) - np.identity(U.shape[0]))
prog.print()