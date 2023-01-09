import strawberryfields as sf
from strawberryfields import ops
import numpy as np
from typing import Tuple, List
import itertools as iter
from utils import bitstring_to_int
from itertools import combinations


class GBS_simulation:

    def get_fock_prob(
        self,
        state_vec: np.ndarray,
        modes: List, 
        photon_numbers: Tuple[int, ...]
    ) -> float:
        return np.sum([(x*x.conjugate()).real for i, x in np.ndenumerate(state_vec) if tuple([i[j] for j in modes]) == photon_numbers])

    def convert_to_clicks(self, outcomes: List) -> List:
        """Converts list of photon number patterns into click
        patterns i.e. only distinguish between detection or no
        detection."""
        mutable_outcomes = [list(y) for y in outcomes]
        for outcome in mutable_outcomes:
            for i, x in enumerate(outcome):
                if x > 0:
                    outcome[i] = 1
        return [tuple(y) for y in mutable_outcomes]

    def get_ideal_marginal(
        self,
        n_modes: int,
        fock_cutoff: int,
        squeezing_params: np.ndarray,
        unitary: np.ndarray,
        target_modes: List
    ) -> np.ndarray:
        prog = sf.Program(n_modes)
        with prog.context as q:
            for i, s in enumerate(squeezing_params):
                ops.Sgate(s) | q[i]
            ops.Interferometer(unitary) | q
        eng = sf.Engine("fock", backend_options={"cutoff_dim": fock_cutoff})
        result = eng.run(prog)
        fock_ket = result.state.ket()
        print(f'Sum of all fock probabilities for cutoff {fock_cutoff}:', np.sum(result.state.all_fock_probs()))
        outcomes = [p for p in iter.product(list(range(fock_cutoff)), repeat = len(target_modes))]
        marginal = [self.get_fock_prob(fock_ket, target_modes, n) for n in outcomes]
        clicks = [bitstring_to_int(x) for x in self.convert_to_clicks(outcomes)]
        inds_to_sum = [[i for i, x in enumerate(clicks) if x == j] for j in range(2**len(target_modes))]
        threshold_marginal = [np.sum([p for i, p in enumerate(marginal) if i in inds]) for inds in inds_to_sum]
        return threshold_marginal
    
    def get_noisy_marginal(
        self,
        n_modes: int,
        fock_cutoff: int,
        squeezing_params: np.ndarray,
        unitary: np.ndarray,
        target_modes: List,
        theta: float = np.pi/4
    ) -> np.ndarray:
        prog = sf.Program(2*n_modes)
        with prog.context as q:
            for i, s in enumerate(squeezing_params):
                ops.Sgate(s) | q[i]
            for cmd in ops.Interferometer(unitary).decompose(tuple([qubit for qubit in q[:n_modes]])):
                cmd.op | cmd.reg
            for i, qubit in enumerate(q[:n_modes]):
                ops.BSgate(theta) | (qubit, q[n_modes + i]) 
        eng = sf.Engine("fock", backend_options={"cutoff_dim": fock_cutoff})
        result = eng.run(prog)
        fock_ket = result.state.ket()
        #print(f'Sum of all fock probabilities for cutoff {fock_cutoff}:', np.sum(result.state.all_fock_probs()))
        outcomes = [p for p in iter.product(list(range(fock_cutoff)), repeat = len(target_modes))]
        marginal = [self.get_fock_prob(fock_ket, target_modes, n) for n in outcomes]
        clicks = [bitstring_to_int(x) for x in self.convert_to_clicks(outcomes)]
        inds_to_sum = [[i for i, x in enumerate(clicks) if x == j] for j in range(2**len(target_modes))]
        threshold_marginal = [np.sum([p for i, p in enumerate(marginal) if i in inds]) for inds in inds_to_sum]
        return threshold_marginal

    def get_ideal_marginals_from_simulation(
        self,
        n_modes: int,
        fock_cutoff: int,
        squeezing_params: np.ndarray,
        unitary: np.ndarray,
        k_order: int,
        format: str = 'No'
    ) -> np.ndarray:
        """Gets ground truth k-th order marginals from the output statevector of the Strawberry
        Fields simulation of a GBS experiment with the given number of modes, squeezing
        parameters and unitary matrix (defining the interferometer). The fock cutoff defines
        the truncation of the fock basis in the simulation."""
        comb = [list(c) for c in combinations(list(range(n_modes)), k_order)]
        marginals : List = []
        for modes in comb:
            marg = self.get_ideal_marginal(n_modes, fock_cutoff, squeezing_params, unitary, modes)
            marginals.append([modes, marg])
        if format == 'No':
            return marginals
        else:
            return self.format_marginals(marginals, n_modes)
    
    def get_noisy_marginals_from_simulation(
        self,
        n_modes: int,
        fock_cutoff: int,
        squeezing_params: np.ndarray,
        unitary: np.ndarray,
        k_order: int,
        theta: float = np.pi/4,
        format: str = 'No'
    ) -> np.ndarray:
        """Gets ground truth k-th order marginals from the output statevector of the Strawberry
        Fields simulation of a GBS experiment with the given number of modes, squeezing
        parameters and unitary matrix (defining the interferometer). The fock cutoff defines
        the truncation of the fock basis in the simulation."""
        comb = [list(c) for c in combinations(list(range(n_modes)), k_order)]
        marginals : List = []
        for modes in comb:
            marg = self.get_noisy_marginal(n_modes, fock_cutoff, squeezing_params, unitary, modes, theta)
            marginals.append([modes, marg])
        if format == 'No':
            return marginals
        else:
            return self.format_marginals(marginals, n_modes)
    
    def format_marginals(self, marginals: List, n_modes: int) -> List:
        """Format ground-truth marginals so that they can be used as inputs of the 
        greedy algorithm."""
        k_order = len(marginals[0][0])
        formatted_marg : List = []
        for j in range(k_order - 1, n_modes):
            to_join: List = []
            for elem in marginals:
                last_mode_index = elem[0][-1]
                if last_mode_index == j:
                    to_join.append(elem)
            formatted_marg.append(to_join)
        return formatted_marg
        
