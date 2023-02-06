from typing import List
import strawberryfields as sf
from strawberryfields import ops
import numpy as np

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
    std: float
):
    return none