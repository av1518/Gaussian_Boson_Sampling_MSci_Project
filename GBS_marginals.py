from typing import List, Tuple
import numpy as np
import copy 
from thewalrus import tor, hafnian
from utils import get_click_indices, get_binary_basis
import strawberryfields as sf
import strawberryfields.ops as ops
import thewalrus

class Marginal:

    def Ch(self, r: float) -> np.ndarray:
        """Returns Ch submatrix of the squeezing vector."""
        return np.array([[np.cosh(r), 0], [0,np.cosh(r)]])

    def Sh(self, r: float) -> np.ndarray:
        """Returns Sh submatrix of the squeezing vector."""
        return np.array([[np.sinh(r), 0], [0,np.sinh(r)]])

    def get_two_mode_squeezing_S(self, r_k: np.ndarray) -> np.ndarray:
        """Returns the squeezing matrix for the two-mode squeezing parameters
        r_k (following the construction of Google's paper)."""
        N = len(r_k)*2
        S_ch = np.zeros((N,N))
        S_sh = np.zeros((N,N))
        for count,value in enumerate(r_k):
            i = 2 * count
            S_ch[i:i+2, i:i+2] = self.Ch(value)
            S_sh[i:i+2, i:i+2] = self.Sh(value)
        S_firstcolumn = np.concatenate((S_ch,S_sh), axis = 0)
        S_secondcolumn = np.concatenate((S_sh,S_ch), axis = 0)
        S = np.concatenate((S_firstcolumn,S_secondcolumn),axis = 1)
        return S
    
    def get_S(self, r_k: np.ndarray) -> np.ndarray:
        """Returns the squeezing matrix for the single-mode squeezing parameters
        r_k (following the construction of original GBS paper)."""
        N = len(r_k)
        S_ch = np.zeros((N,N), dtype = complex)
        S_sh = np.zeros((N,N), dtype = complex)
        for i, r in enumerate(r_k):
            S_ch[i,i] = np.cosh(r)
            S_sh[i,i] = np.sinh(r)
        S_firstcolumn = np.concatenate((S_ch, S_sh), axis = 0)
        S_secondcolumn = np.concatenate((S_sh, S_ch), axis = 0)
        S = np.concatenate((S_firstcolumn, S_secondcolumn), axis = 1)
        return S

    def get_input_covariance_matrix(self, S: np.ndarray) -> np.ndarray:
        """Returns the matrix sigma_in needed to obtain the covariance matrix."""
        sigma_vac = 0.5*np.identity(len(S))
        return np.dot(S, np.dot(sigma_vac, S.T)) #S.T should have been S dagger, for real S
        #For real squezing params, S == S.T
        
    def get_output_covariance_matrix(self, matrix: np.ndarray, r_k: np.ndarray) -> np.ndarray:
        """Returns the covariance matrix of a GBS experiment defined by an interferometer
        matrix (the transformation matrix T or an ideal unitary matrix) and the squeezing
        parameters r_k."""
        sigma_in = self.get_input_covariance_matrix(self.get_S(r_k))
        m_len, m_height = np.shape(matrix)
        m_transpose_len, m_transpose_height = np.shape(matrix.T)
        first = np.zeros((m_len * 2, m_height * 2), dtype = complex)
        second = first.T
        first[0:m_len, 0:m_height] = matrix
        first[m_len:, m_height:] = matrix.conjugate()
        second[0:m_transpose_len, 0:m_transpose_height] = matrix.T.conjugate()
        second[m_transpose_len:, m_transpose_height:] = matrix.T
        #check if T is unitary, if not -> google paper's way
        if np.allclose(matrix.dot(matrix.conj().T), np.identity(matrix.shape[0]), atol=1e-12):
            return np.dot(first, np.dot(sigma_in, second))
        else:
            #google equation
            return np.identity(m_len*2) - 1/2 * np.dot(first, second) + np.dot(first,np.dot(sigma_in,second)) 

    def get_reduced_matrix(self, sigma: np.ndarray, R: List) -> np.ndarray:
        '''
        Parameters
        ----------
        sigma: covariance matrix
        R: mode indices considered

        Returns the reduced matrix associated with the input mode indices.
        '''
        indices = copy.deepcopy(R)
        for i in R:         
            indices.append(int(i + len(sigma)/2))
        sigma_red = np.empty((len(indices),len(indices)), dtype=complex)
        for r_index, row in enumerate(indices):
            for c_index, column in enumerate(indices):
                sigma_red[r_index, c_index] = sigma[row, column]
        return sigma_red

    def get_reduced_B(self, sigma: np.ndarray, R: List) -> np.ndarray:
        '''
        Parameters
        ----------
        sigma: covariance matrix
        R: mode indices considered

        Returns the reduced matrix associated with the input mode indices.
        '''
        indices = copy.deepcopy(R)
        sigma_red = np.empty((len(indices),len(indices)), dtype=complex)
        for r_index, row in enumerate(indices):
            for c_index, column in enumerate(indices):
                sigma_red[r_index, c_index] = sigma[row, column]
        return sigma_red

    def get_prob_all_zero_bitstring(self, cov_matrix: np.ndarray) -> float:
        """Calculates the probability of the outcome with no photon detections from
        the overlap integral between the Wigner function of the output state of the
        GBS experiment (squeezed coherent) and the Wigner function of the vacuum state. 
        The covariance matrix of the vacuum state is the identity, and the product of
        two gaussians results in a gaussian with a covariance matrix which is of the
        form of the variable new_cov_matrix defined below."""
        dim = cov_matrix.shape[0]
        new_cov_matrix = 2*np.dot(cov_matrix, np.linalg.inv(cov_matrix + 0.5*np.identity(dim)))
        factor = 1/(4*(np.pi**2)*np.sqrt(0.5*np.linalg.det(cov_matrix)))
        return factor*np.sqrt(np.linalg.det(2*np.pi*new_cov_matrix))
    
    def get_B_matrix(
        self, 
        unitary: np.ndarray,
        squeezing_params: np.ndarray
    ) -> np.ndarray:
        """Returns the B matrix outlined in the paper 'Gaussian Boson Sampling.'
        This matrix defines a GBS experiment with a given unitary and squeezing 
        paramters."""
        N = len(squeezing_params)
        S = np.zeros((N,N), dtype=complex)
        for i, r in enumerate(squeezing_params):
            S[i,i] = np.tanh(r)
        return np.dot(unitary, np.dot(S, unitary.T))

    def get_A_matrix(
        self,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """Returns the A matrix outlined in Kolthammer's paper 'Experimentally
        finding dense subgraphs using a time-bin encoded GBS device.' This matrix
        is obtained from the covariance matrix of the output state of a GBS
        experiment."""
        dim = len(cov_matrix)
        X_matrix = np.zeros((dim, dim), dtype = complex)
        X_matrix[0:int(dim/2), int(dim/2):] = np.identity(int(dim/2))
        X_matrix[int(dim/2):, 0:int(dim/2)] = np.identity(int(dim/2))
        Q_matrix = cov_matrix + np.identity(dim)/2
        inverse_Q_matrix = np.linalg.inv(Q_matrix)
        term = np.identity(dim) - inverse_Q_matrix
        return np.dot(X_matrix, term)

    def get_single_outcome_probability_tor(
        self,
        bitstring: Tuple,
        sigma: np.ndarray
    ) -> float:
        """Return probability of a single output detection pattern
        in a GBS experiment defined by the squeezing parameters and
        the transformation matrix (which determine sigma)."""
        set_S = get_click_indices(bitstring)
        if not set_S:
            cov_matrix = thewalrus.quantum.Covmat(sigma)
            return self.get_prob_all_zero_bitstring(cov_matrix)
        else:
            sigma_inv_reduced = self.get_reduced_matrix(np.linalg.inv(sigma), set_S)
            O_s = np.identity(len(sigma_inv_reduced)) - sigma_inv_reduced
            return tor(O_s) / np.sqrt(np.linalg.det(sigma))
    
    def get_single_outcome_probability_haf(
        self,
        bitstring: Tuple,
        sigma: np.ndarray, 
        A_matrix: np.ndarray
    ) -> float:
        """Return probability of a single output detection pattern
        in a GBS experiment defined by the squeezing parameters and
        the transformation matrix (which determine sigma)."""
        set_S = get_click_indices(bitstring)
        if not set_S:
            return self.get_prob_all_zero_bitstring(sigma)
        else:
            A_s = self.get_reduced_matrix(A_matrix, set_S)
            haf = hafnian(A_s)
            return haf / np.sqrt(np.linalg.det(sigma))

    def get_marginal_distribution_from_tor(
        self,
        mode_indices: List,
        interferometer_matrix: np.ndarray,
        r_k: np.ndarray
    ) -> np.ndarray:
        """Returns marginal distribution of the specified modes."""
        cov_matrix = thewalrus.quantum.Qmat(self.get_cov_matrix_sf(interferometer_matrix, r_k))
        k_order = len(mode_indices)
        binary_basis = get_binary_basis(k_order)
        reduced_sigma = self.get_reduced_matrix(cov_matrix, mode_indices)
        distr = [self.get_single_outcome_probability_tor(string, reduced_sigma).real for string in binary_basis]
        return np.array(distr)
    
    def get_marginal_distribution_from_haf(
        self,
        mode_indices: List,
        interferometer_matrix: np.ndarray,
        r_k: np.ndarray
    ) -> np.ndarray:
        """Returns marginal distribution of the specified modes."""
        cov_matrix = self.get_output_covariance_matrix(interferometer_matrix, r_k)
        k_order = len(mode_indices)
        binary_basis = get_binary_basis(k_order)
        reduced_sigma = self.get_reduced_matrix(cov_matrix, mode_indices)
        B_matrix = self.get_B_matrix(interferometer_matrix, r_k)
        dim = len(B_matrix)
        A_matrix = np.zeros((2*dim, 2*dim), dtype=complex)
        A_matrix[0:dim, 0:dim] = B_matrix
        A_matrix[dim:, dim:] = B_matrix.conjugate()
        distr = [self.get_single_outcome_probability_haf(string, reduced_sigma, A_matrix).real for string in binary_basis]
        return np.array(distr)

    def get_cov_matrix_sf(self,
        U: np.ndarray, 
        r_k: np.ndarray,
    ) -> np.ndarray:
        '''Returns covariance matrix using sf built in method (builds
        the circuit first)'''
        if len(r_k) != len(U):
            raise Exception('r_k and U must have the same length')
        n_modes = len(r_k)
        p = sf.Program(n_modes)
        with p.context as q:
            for i, r in enumerate(r_k):
                ops.Sgate(r) | q[i]
            ops.Interferometer(U) | q
        e = sf.Engine(backend = "gaussian")
        state = e.run(p).state
        sigma = state.cov()
        if thewalrus.quantum.is_valid_cov(sigma) == True:
            return sigma
        else:
            raise Exception('Covariance matrix not valid')



    

    
    