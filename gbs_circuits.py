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
    """Returns Strawberry Fields program corresponding to a GBS circuit with
    the given number of modes, squeezing parameters (applied to all of the
    modes), and an interferometer (defined by the unitary)."""
    prog = sf.Program(n_modes)
    with prog.context as q:
        for i, s in enumerate(squeezing_params):
            ops.Sgate(s) | q[i]
        ops.Interferometer(unitary) | q
    return prog

def get_gbs_circuit_with_loss_channel(
    n_modes: int, 
    squeezing_params: List, 
    unitary: np.ndarray,
    loss: float
):
    """Returns Strawberry Fields program corresponding to a GBS circuit with
    the given number of modes, squeezing parameters (applied to all of the
    modes), and an interferometer (defined by the unitary)."""
    prog = sf.Program(n_modes)
    with prog.context as q:
        for i, s in enumerate(squeezing_params):
            ops.Sgate(s) | q[i]
        ops.Interferometer(unitary) | q
        for qubit in q:
            ops.LossChannel(loss) | qubit
    return prog

def get_gbs_circuit_with_optical_loss(
    n_modes: int, 
    squeezing_params: List, 
    unitary: np.ndarray, 
    loss: float
):
    """Returns Strawberry Fields program corresponding to a GBS circuit with
    then same construction as get_ideal_gbs_circuit but with a beamsplitter 
    at the end of each mode (connecting them to some extra vaccuum modes) so
    that there is some probability that the photon is lost. The loss parameter
    specifies this probability (0 for no loss and 1 for 100% loss). All of the
    beamsplitters have the same loss parameter."""
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

def get_gbs_circuit_with_distinguishable_photons(
    n_modes: int,
    unitary: np.ndarray,
    squeezing_main: List,
    squeezing_imperfection: float
):
    """Returns a list of Strawberry Fields programs. The first program
    of the list is the ideal GBS circuit determined by the unitary and
    the squeezing_main parameters. The other programs are copies of the
    same circuit but with a squeezing gate (with the squeezing_imperfection
    parameter) in only one mode. The rest of the modes are initialised in
    vaccuum. The squeezing_main parameters are all equal to each other."""
    progs = [get_ideal_gbs_circuit(n_modes, squeezing_main, unitary)]
    for j in range(n_modes):
        prog = sf.Program(n_modes)
        with prog.context as q:
            ops.Sgate(squeezing_imperfection) | q[j]
            ops.Interferometer(unitary) | q
        progs.append(prog)
    return progs
