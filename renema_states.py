#%%
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from greedy import Greedy
from scipy.stats import unitary_group
from gbs_simulation import GBS_simulation
import strawberryfields as sf
from strawberryfields import ops
from utils import total_variation_distance
from tqdm import tqdm
from utils import bitstring_to_int, convert_to_clicks
import itertools as iter
from itertools import combinations

n_modes = 4
cutoff = 6
k_order = 2
L = 500
unitary = unitary_group.rvs(n_modes)

def ideal_renema_circuit_10(n_modes, unitary):
    prog = sf.Program(n_modes)
    with prog.context as q:
        for i in range(int(np.ceil(0.5*n_modes))):
            ops.Fock(1) | q[i]
        ops.Interferometer(unitary) | q
    return prog

def ideal_renema_circuit_01(n_modes, unitary):
    prog = sf.Program(n_modes)
    with prog.context as q:
        for i in range(int(np.ceil(0.5*n_modes)), n_modes):
            ops.Fock(1) | q[i]
        ops.Interferometer(unitary) | q
    return prog

def noisy_renema_circuit_10(n_modes, unitary, loss):
    prog = sf.Program(2*n_modes)
    with prog.context as q:
        for i in range(int(np.ceil(0.5*n_modes))):
            ops.Fock(1) | q[i]
        for cmd in ops.Interferometer(unitary).decompose(tuple([qubit for qubit in q[:n_modes]])):
            cmd.op | cmd.reg
        for i, qubit in enumerate(q[:n_modes]):
            ops.BSgate(loss) | (qubit, q[n_modes + i])
    return prog

def noisy_renema_circuit_01(n_modes, unitary, loss):
    prog = sf.Program(2*n_modes)
    with prog.context as q:
        for i in range(int(np.ceil(0.5*n_modes)), n_modes):
            ops.Fock(1) | q[i]
        for cmd in ops.Interferometer(unitary).decompose(tuple([qubit for qubit in q[:n_modes]])):
            cmd.op | cmd.reg
        for i, qubit in enumerate(q[:n_modes]):
            ops.BSgate(loss) | (qubit, q[n_modes + i])
    return prog

def get_state_vector_from_program(
    program,
    fock_cutoff: int
) -> np.ndarray:
    """Runs a Strawberry Fields program in the Fock backend (with the specified cutoff)
    and obtains the threshold marginal distribution of the specified target modes."""
    eng = sf.Engine("fock", backend_options={"cutoff_dim": fock_cutoff})
    result = eng.run(program)
    print(result.state)
    fock_ket = result.state.ket()
    return fock_ket

def get_fock_prob(
    state_vec: np.ndarray,
    modes: List, 
    photon_numbers: Tuple[int, ...]
) -> float:
    """Returns the probability of detecting a specific photon pattern in the specified
    modes from the output state vector of a GBS simulation."""
    return np.sum([(x*x.conjugate()).real for i, x in np.ndenumerate(state_vec) if tuple([i[j] for j in modes]) == photon_numbers])

def get_threshold_marginal_from_statevec(
    ket: np.ndarray,
    target_modes: List
) -> List:
    fock_cutoff = ket.shape[0]
    outcomes = [p for p in iter.product(list(range(fock_cutoff)), repeat = len(target_modes))]
    marginal = [get_fock_prob(ket, target_modes, n) for n in outcomes]
    clicks = [bitstring_to_int(x) for x in convert_to_clicks(outcomes)]
    inds_to_sum = [[i for i, x in enumerate(clicks) if x == j] for j in range(2**len(target_modes))]
    threshold_marginal = [np.sum([p for i, p in enumerate(marginal) if i in inds]) for inds in inds_to_sum]
    return threshold_marginal

def get_all_ideal_marginals_from_statevec(
    n_modes: int,
    ket: np.ndarray,
    k_order: int
) -> np.ndarray:
    comb = [list(c) for c in combinations(list(range(n_modes)), k_order)]
    marginals : List = []
    for modes in comb:
        marg = get_threshold_marginal_from_statevec(ket, modes)
        marginals.append([modes, marg])
    return np.array(marginals)

def get_renema_output_statevec(n_modes, unitary, cutoff):
    ket1 = get_state_vector_from_program(ideal_renema_circuit_01(n_modes, unitary), cutoff)
    ket2 = get_state_vector_from_program(ideal_renema_circuit_10(n_modes, unitary), cutoff)
    superposition_ket = (1/np.sqrt(2))*(ket1 + ket2)
    return superposition_ket

def get_noisy_renema_output_statevec(n_modes, unitary, cutoff, loss):
    ket1 = get_state_vector_from_program(noisy_renema_circuit_01(n_modes, unitary), cutoff, loss)
    ket2 = get_state_vector_from_program(noisy_renema_circuit_10(n_modes, unitary), cutoff, loss)
    superposition_ket = (1/np.sqrt(2))*(ket1 + ket2)
    return superposition_ket

#%%
gbs = GBS_simulation()
greedy = Greedy()

ideal_margs = get_all_ideal_marginals_from_statevec(n_modes, get_renema_output_statevec(n_modes, unitary, cutoff), k_order)
greedy_matrix = greedy.get_S_matrix(n_modes, L, k_order, ideal_margs)
greedy_distr = greedy.get_distribution_from_outcomes(greedy_matrix)

#%%
loss = np.linspace(0, 1, 15 )

distances = []
for i in tqdm(loss):
    ket = get_noisy_renema_output_statevec(n_modes, unitary, cutoff, i)
    ideal_distr = get_threshold_marginal_from_statevec(ket, list(range(n_modes)))
    distance = total_variation_distance(ideal_distr, greedy_distr)
    distances.append(distance)

plt.plot(loss, distances, 'o-', label = f'modes = {n_modes}, cutoff = {cutoff} ')
plt.xlabel(r'Noise ($\phi$) ')
plt.ylabel(r'$\mathcal{D}$(Greedy,Ground)')
plt.legend()
plt.show()

