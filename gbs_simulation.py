import strawberryfields as sf
from strawberryfields import ops
import numpy as np
from typing import Tuple, List
import itertools as iter
from utils import bitstring_to_int, convert_to_clicks
from itertools import combinations
from gbs_circuits import get_ideal_gbs_circuit, get_gbs_circuit_with_optical_loss
import qutip

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
        print(f'Sum of all fock probabilities for cutoff {fock_cutoff}:', np.sum(result.state.all_fock_probs()))
        outcomes = [p for p in iter.product(list(range(fock_cutoff)), repeat = len(target_modes))]
        marginal = [self.get_fock_prob(fock_ket, target_modes, n) for n in outcomes]
        clicks = [bitstring_to_int(x) for x in convert_to_clicks(outcomes)]
        inds_to_sum = [[i for i, x in enumerate(clicks) if x == j] for j in range(2**len(target_modes))]
        threshold_marginal = [np.sum([p for i, p in enumerate(marginal) if i in inds]) for inds in inds_to_sum]
        return threshold_marginal

    def get_ideal_marginal_from_simulation(
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
        prog = get_ideal_gbs_circuit(n_modes, squeezing_params, unitary)
        return self._get_threshold_marginal_from_program(prog, target_modes, fock_cutoff)
    
    def get_noisy_marginal_from_simulation(
        self,
        n_modes: int,
        fock_cutoff: int,
        squeezing_params: np.ndarray,
        unitary: np.ndarray,
        target_modes: List,
        loss: float = 0.5
    ) -> List:
        """Returns the marginal distribution of the target modes in a GBS simulation
        (incorporating optical loss) parameterised by the squeezing parameters, the
        interferometer unitary, the fock cut-off value and the number of modes. The loss
        factor goes from 0 (no loss) to 1 (maximum loss)."""
        loss = loss*np.pi/2 # convert loss to beamsplitter angle (pi/2 is max loss)
        prog = get_gbs_circuit_with_optical_loss(n_modes, squeezing_params, unitary, loss)
        return self._get_threshold_marginal_from_program(prog, target_modes, fock_cutoff)

    def get_all_ideal_marginals_from_simulation(
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
        the truncation of the fock basis in the simulation. Returns an array where each element
        has two sublists. The first one is the set of mode indices of that marginal, and the 
        second one is the marginal distribution."""
        comb = [list(c) for c in combinations(list(range(n_modes)), k_order)]
        marginals : List = []
        for modes in comb:
            marg = self.get_ideal_marginal_from_simulation(n_modes, fock_cutoff, squeezing_params, unitary, modes)
            marginals.append([modes, marg])
        return np.array(marginals)
    
    def get_all_noisy_marginals_from_simulation(
        self,
        n_modes: int,
        fock_cutoff: int,
        squeezing_params: np.ndarray,
        unitary: np.ndarray,
        k_order: int,
        loss: float = 0.5
    ) -> np.ndarray:
        """Gets ground truth k-th order marginals from the output statevector of the Strawberry
        Fields simulation of a GBS experiment (incorporating optical loss) with the given number
        of modes, squeezing parameters and the interferometer unitary. The fock cutoff defines
        the truncation of the fock basis in the simulation. Returns an array where each element
        has two sublists. The first one is the set of mode indices of that marginal, and the 
        second one is the marginal distribution. The loss factor goes from 0 (no loss) to pi/2
        (maximum loss)."""
        comb = [list(c) for c in combinations(list(range(n_modes)), k_order)]
        marginals : List = []
        for modes in comb:
            marg = self.get_noisy_marginal_from_simulation(n_modes, fock_cutoff, squeezing_params, unitary, modes, loss)
            marginals.append([modes, marg])
        return np.array(marginals)
    
    def turn_detections_into_projection_operators(
        self, 
        outcome: Tuple, 
        cutoff: int
    ) -> np.ndarray:
        """Turn photon-detection patterns into their corresponding
        projection operators."""
        single_mode_states = []
        for detection in outcome:
            vector = np.zeros(cutoff)
            vector[detection-1] = 1
            outer_prod = np.outer(vector, vector)
            single_mode_states.append(outer_prod)
        operator = single_mode_states[0]
        if len(single_mode_states) > 1:
            for i in range(1, len(single_mode_states)):
                operator = np.tensordot(operator, single_mode_states[i], 0)
        return operator
    
    def turn_detections_into_product_of_density_matrices(
        self, 
        outcome: Tuple, 
        cutoff: int
    ) -> np.ndarray:
        """Turn photon-detection patterns into their corresponding
        projection operators."""
        single_mode_states = []
        for detection in outcome:
            vector = np.zeros(cutoff)
            vector[detection-1] = 1
            outer_prod = np.outer(vector, vector)
            single_mode_states.append(outer_prod)
        return single_mode_states
    
    def trace(self, tensor: np.ndarray) -> float:
        """Returns trace of an operator."""
        shape = tensor.shape
        suma = np.trace(tensor, axis1= len(shape)-2, axis2=len(shape)-1)
        shape = suma.shape
        for i in range(len(shape)-1, -1, -2):
            if i-1 >= 0:
                suma = np.trace(suma, axis1=i-1, axis2=i)
        return suma
    
    def get_noisy_marginal_from_bosonic_simulation(
        self,
        n_modes: int,
        squeezing_params: np.ndarray,
        unitary: np.ndarray,
        target_modes: List,
        loss: float = 0.5
    ) -> List:
        """Returns the marginal distribution of the target modes in a GBS simulation
        (incorporating optical loss) parameterised by the squeezing parameters, the
        interferometer unitary, the fock cut-off value and the number of modes."""
        prog = get_gbs_circuit_with_optical_loss(n_modes, squeezing_params, unitary, loss) 
        eng = sf.Engine("bosonic")
        result = eng.run(prog).state
        reduced_dm = result.reduced_dm(target_modes)
        cutoff = len(reduced_dm[0])
        outcomes = [p for p in iter.product(list(range(cutoff)), repeat = len(target_modes))]
        operators = [self.turn_detections_into_projection_operators(i, cutoff) for i in outcomes]
        marginal = [self.trace(np.tensordot(P, reduced_dm).real) for P in operators]
        print(marginal)
        clicks = [bitstring_to_int(x) for x in convert_to_clicks(outcomes)]
        inds_to_sum = [[i for i, x in enumerate(clicks) if x == j] for j in range(2**len(target_modes))]
        threshold_marginal = [np.sum([p for i, p in enumerate(marginal) if i in inds]) for inds in inds_to_sum]
        return threshold_marginal
    
    def get_noisy_marginal_from_bosonic_simulation2(
        self,
        n_modes: int,
        squeezing_params: np.ndarray,
        unitary: np.ndarray,
        target_modes: List,
        loss: float = 0.5
    ) -> List:
        """Returns the marginal distribution of the target modes in a GBS simulation
        (incorporating optical loss) parameterised by the squeezing parameters, the
        interferometer unitary, the fock cut-off value and the number of modes."""
        prog = get_gbs_circuit_with_optical_loss(n_modes, squeezing_params, unitary, loss) 
        eng = sf.Engine("bosonic")
        result = eng.run(prog).state
        reduced_dm = result.reduced_dm(target_modes)
        cutoff = len(reduced_dm[0])
        outcomes = [p for p in iter.product(list(range(cutoff)), repeat = len(target_modes))]
        operators = [self.turn_detections_into_projection_operators(i, cutoff) for i in outcomes]
        marginal = [qutip.measurement.measurement_statistics_observable(qutip.Qobj(reduced_dm), qutip.Qobj(P))[2] for P in operators]
        print(marginal)
        clicks = [bitstring_to_int(x) for x in convert_to_clicks(outcomes)]
        inds_to_sum = [[i for i, x in enumerate(clicks) if x == j] for j in range(2**len(target_modes))]
        threshold_marginal = [np.sum([p for i, p in enumerate(marginal) if i in inds]) for inds in inds_to_sum]
        return threshold_marginal

