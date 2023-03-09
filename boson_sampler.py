import strawberryfields as sf
import numpy as np
from typing import Tuple, List
import itertools as iter
from utils import bitstring_to_int, convert_to_clicks, total_variation_distance
from itertools import combinations
from strawberryfields import ops
from greedy import Greedy
from scipy.stats import unitary_group


class BS_simulation:

    def get_fock_prob(
        self,
        state_vec: np.ndarray,
        modes: List, 
        photon_numbers: Tuple[int, ...]
    ) -> float:
        """Returns the probability of detecting a specific photon pattern in the specified
        modes from the output state vector of a GBS simulation."""
        return np.sum([(x*x.conjugate()).real for i, x in np.ndenumerate(state_vec) if tuple([i[j] for j in modes]) == photon_numbers])

    def get_threshold_marginal_from_statevec(
        self,
        program,
        target_modes: List,
        fock_cutoff: int
    ) -> List:
        """Runs a Strawberry Fields program in the Fock backend (with the specified cutoff)
        and obtains the threshold marginal distribution of the specified target modes."""
        eng = sf.Engine("fock", backend_options={"cutoff_dim": fock_cutoff})
        result = eng.run(program)
        # print('Number expectation:', result.state.number_expectation(target_modes)[0])
        fock_ket = result.state.ket()
        # print(f'Sum of all fock probabilities for cutoff {fock_cutoff}:', np.sum(result.state.all_fock_probs()))
        outcomes = [p for p in iter.product(list(range(fock_cutoff)), repeat = len(target_modes))]
        marginal = [self.get_fock_prob(fock_ket, target_modes, n) for n in outcomes]
        clicks = [bitstring_to_int(x) for x in convert_to_clicks(outcomes)]
        inds_to_sum = [[i for i, x in enumerate(clicks) if x == j] for j in range(2**len(target_modes))]
        threshold_marginal = [np.sum([p for i, p in enumerate(marginal) if i in inds]) for inds in inds_to_sum]
        return threshold_marginal
    
    def get_threshold_marginal_fock_backend(
        self,
        program,
        target_modes: List,
        fock_cutoff: int
    ) -> List:
        """Runs a Strawberry Fields program in the Fock backend (with the specified cutoff)
        and obtains the threshold marginal distribution of the specified target modes."""
        n_modes = program.num_subsystems
        eng = sf.Engine("fock", backend_options={"cutoff_dim": fock_cutoff})
        result = eng.run(program)
        probs = result.state.all_fock_probs()
        print(np.sum(probs))
        inds = tuple([i for i in range(n_modes) if i not in target_modes])
        outcomes = [p for p in iter.product(list(range(fock_cutoff)), repeat = len(target_modes))]
        marginal = np.sum(probs, axis=inds)
        marginal = [marginal[i] for i in outcomes]
        clicks = [bitstring_to_int(x) for x in convert_to_clicks(outcomes)]
        inds_to_sum = [[i for i, x in enumerate(clicks) if x == j] for j in range(2**len(target_modes))]
        threshold_marginal = [np.sum([p for i, p in enumerate(marginal) if i in inds]) for inds in inds_to_sum]
        return threshold_marginal

    def get_ideal_BS_marginal_from_fock_simulation(
        self,
        n_modes: int,
        fock_cutoff: int,
        n_input_photons: int,
        unitary: np.ndarray,
        target_modes: List
    ) -> List:
        """Returns the marginal distribution of the target modes in a GBS simulation
        parameterised by the squeezing parameters, the interferometer unitary, the
        fock cut-off value and the number of modes."""
        index = np.zeros((n_modes,), dtype=np.int16)
        index[:n_input_photons] = 1
        ket = np.zeros([fock_cutoff]*n_modes, dtype=np.complex128)
        ket[tuple(index)] = 1.0 + 1j*0.0
        prog = sf.Program(n_modes)
        with prog.context as q:
            ops.Ket(ket) | q
            ops.Interferometer(unitary) | q
        return self.get_threshold_marginal_fock_backend(prog, target_modes, fock_cutoff)
    
    def get_all_ideal_marginals_from_fock_simulation(
        self,
        n_modes: int,
        fock_cutoff: int,
        n_input_photons: int,
        unitary: np.ndarray,
        k_order: int
    ) -> np.ndarray:
        """Gets ground truth k-th order marginals from the output statevector of the Strawberry
        Fields simulation of a GBS experiment with the given number of modes, squeezing
        parameters and unitary matrix (defining the interferometer). The fock cutoff defines
        the truncation of the fock basis in the simulation. Returns an array where each element
        has two sublists. The first one is the set of mode indices of that marginal, and the 
        second one is the marginal distribution."""
        comb = [list(c) for c in combinations(list(range(n_modes)), k_order)]
        marginals : List = []
        for modes in comb:
            marg = self.get_ideal_BS_marginal_from_fock_simulation(n_modes, fock_cutoff, n_input_photons, unitary, modes)
            marginals.append([modes, marg])
        return np.array(marginals)
    

bs = BS_simulation()
greedy = Greedy()

n_modes = 6
cutoff = 5
k_order = 2
L = 1200
unitary = unitary_group.rvs(n_modes, random_state=1)
n_input = 3

ideal_margs = bs.get_all_ideal_marginals_from_fock_simulation(n_modes, cutoff, n_input, unitary, k_order)
greedy_matrix = greedy.get_S_matrix(n_modes, L, k_order, ideal_margs)
greedy_distr = greedy.get_distribution_from_outcomes(greedy_matrix)
np.save(f'greedy_bs_distr_n={n_modes}_cut={cutoff}_L={L}_n_input={n_input}', greedy_distr)
ideal_distr = np.array(bs.get_ideal_BS_marginal_from_fock_simulation(n_modes, cutoff, n_input, unitary, list(range(n_modes))))
np.save(f'ideal_bs_distr_n={n_modes}_cut={cutoff}_n_input={n_input}', ideal_distr)
distance = total_variation_distance(ideal_distr, greedy_distr)
print(distance)