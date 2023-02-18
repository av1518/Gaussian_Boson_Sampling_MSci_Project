from typing import List
import strawberryfields as sf
from strawberryfields import ops
import numpy as np
from utils import complex_to_polar, apply_random_deviation

def get_ideal_gbs_circuit(
    n_modes: int, 
    squeezing_params: List, 
    unitary: np.ndarray, 
):
    prog = sf.Program(n_modes)
    with prog.context as q:
        for i, s in enumerate(squeezing_params):
            ops.Sgate(s) | q[i]
        ops.Interferometer(unitary) | q
    return prog

def get_gbs_circuit_with_optical_loss(
    n_modes: int, 
    squeezing_params: List, 
    unitary: np.ndarray, 
    loss: float
):
    prog = sf.Program(2*n_modes)
    with prog.context as q:
        for i, s in enumerate(squeezing_params):
            ops.Sgate(s) | q[i]
        for cmd in ops.Interferometer(unitary).decompose(tuple([qubit for qubit in q[:n_modes]])):
            cmd.op | cmd.reg
        for i, qubit in enumerate(q[:n_modes]):
            ops.BSgate(loss) | (qubit, q[n_modes + i])
    return prog


def get_gbs_circuit_with_gate_error(
    unitary: np.ndarray,
    squeezing_params: List,
    std: float
):
    '''Takes in unitary, decomposes it using rectangular decomosition into T*V*T_dash (see Kolt's paper)
    then applies random deviations based on normal distribution with given standard deviation on gate
    paramaters, and returns the deviated gate circuit'''
    a = sf.decompositions.rectangular(unitary)
    T = a[0]
    V = complex_to_polar(a[1])
    T_dash = a[2]
    T_noisy = apply_random_deviation(T,std)
    T_dash_noisy = apply_random_deviation(T_dash,std)
    N = T[0][-1] #number of modes
    noisy = sf.Program(N) #noisy gbs program
    eng = sf.Engine("fock", backend_options={"cutoff_dim": 6}) #not sure if this line is needed
    with noisy.context as q:
        for i, s in enumerate(squeezing_params):
            ops.Sgate(s) | q[i]
        for row in T_noisy: #j is a row in T
            ops.Rgate(row[3]) | q[row[0]]
            ops.BSgate(row[2], 0) | (q[row[0]], q[row[1]])
        for j in range(len(V)):
            ops.Rgate(V[j]) | q[j]
        for row in reversed(T_dash_noisy): #take the last row first
            ops.BSgate(-row[2],0) | (q[row[0]], q[row[1]])
            ops.Rgate(-row[3]) | q[row[0]]
    # noisy.compile(compiler="fock").print()
    return noisy
