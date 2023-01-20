import strawberryfields as sf
from strawberryfields import ops
import numpy as np
from typing import Tuple, List
import itertools as iter
from utils import bitstring_to_int, convert_to_clicks
from itertools import combinations


class GBS_simulation:

    def get_fock_prob(
        self,
        state_vec: np.ndarray,
        modes: List, 
        photon_numbers: Tuple[int, ...]
    ) -> float:
        """Returns the probability of detecting a specific photon pattern in the specified
        modes from the output state vector of a GBS simulation."""
        return np.sum([(x*x.conjugate()).real for i, x in np.ndenumerate(state_vec) if tuple([i[j] for j in modes]) == photon_numbers])
    
    def _get_threshold_marginal_from_program(
        self,
        program,
        target_modes: List,
        fock_cutoff: int
    ) -> List:
        """Runs a Strawberry Fields program in the Fock backend (with the specified cutoff)
        and obtains the threshold marginal distribution of the specified target modes."""
        eng = sf.Engine("fock", backend_options={"cutoff_dim": fock_cutoff})
        result = eng.run(program)
        fock_ket = result.state.ket()
        #print(f'Sum of all fock probabilities for cutoff {fock_cutoff}:', np.sum(result.state.all_fock_probs()))
        outcomes = [p for p in iter.product(list(range(fock_cutoff)), repeat = len(target_modes))]
        marginal = [self.get_fock_prob(fock_ket, target_modes, n) for n in outcomes]
        clicks = [bitstring_to_int(x) for x in convert_to_clicks(outcomes)]
        inds_to_sum = [[i for i, x in enumerate(clicks) if x == j] for j in range(2**len(target_modes))]
        threshold_marginal = [np.sum([p for i, p in enumerate(marginal) if i in inds]) for inds in inds_to_sum]
        return threshold_marginal

    def get_ideal_marginal_from_simul(
        self,
        n_modes: int,
        fock_cutoff: int,
        squeezing_params: np.ndarray,
        unitary: np.ndarray,
        target_modes: List
    ) -> List:
        """Returns the marginal distribution of the target modes in a GBS simulation
        parameterised by the squeezing parameters, the interferometer unitary, the
        fock cut-off value and the number of modes."""
        prog = sf.Program(n_modes)
        with prog.context as q:
            for i, s in enumerate(squeezing_params):
                ops.Sgate(s) | q[i]
            ops.Interferometer(unitary) | q
        return self._get_threshold_marginal_from_program(prog, target_modes, fock_cutoff)
    
    def get_noisy_marginal_from_simul(
        self,
        n_modes: int,
        fock_cutoff: int,
        squeezing_params: np.ndarray,
        unitary: np.ndarray,
        target_modes: List,
        theta: float = np.pi/4
    ) -> List:
        """Returns the marginal distribution of the target modes in a GBS simulation
        (incorporating optical loss) parameterised by the squeezing parameters, the
        interferometer unitary, the fock cut-off value and the number of modes."""
        prog = sf.Program(2*n_modes)
        with prog.context as q:
            for i, s in enumerate(squeezing_params):
                ops.Sgate(s) | q[i]
            for cmd in ops.Interferometer(unitary).decompose(tuple([qubit for qubit in q[:n_modes]])):
                cmd.op | cmd.reg
            for i, qubit in enumerate(q[:n_modes]):
                ops.BSgate(theta) | (qubit, q[n_modes + i]) 
        return self._get_threshold_marginal_from_program(prog, target_modes, fock_cutoff)


    def get_ideal_marginals_from_simulation(
        self,
        n_modes: int,
        fock_cutoff: int,
        squeezing_params: np.ndarray,
        unitary: np.ndarray,
        k_order: int
    ) -> np.ndarray:
        """Gets ground truth k-th order marginals from the output statevector of the Strawberry
        Fields simulation of a GBS experiment with the given number of modes, squeezing
        parameters and unitary matrix (defining the interferometer). The fock cutoff defines
        the truncation of the fock basis in the simulation."""
        comb = [list(c) for c in combinations(list(range(n_modes)), k_order)]
        marginals : List = []
        for modes in comb:
            marg = self.get_ideal_marginal_from_simul(n_modes, fock_cutoff, squeezing_params, unitary, modes)
            marginals.append([modes, marg])
        return np.array(marginals)
    
    def get_noisy_marginals_from_simulation(
        self,
        n_modes: int,
        fock_cutoff: int,
        squeezing_params: np.ndarray,
        unitary: np.ndarray,
        k_order: int,
        theta: float = np.pi/4
    ) -> np.ndarray:
        """Gets ground truth k-th order marginals from the output statevector of the Strawberry
        Fields simulation of a GBS experiment (incorporating optical loss) with the given number
        of modes, squeezing parameters and the interferometer unitary. The fock cutoff defines
        the truncation of the fock basis in the simulation."""
        comb = [list(c) for c in combinations(list(range(n_modes)), k_order)]
        marginals : List = []
        for modes in comb:
            marg = self.get_noisy_marginal_from_simul(n_modes, fock_cutoff, squeezing_params, unitary, modes, theta)
            marginals.append([modes, marg])
        return np.array(marginals)
        
