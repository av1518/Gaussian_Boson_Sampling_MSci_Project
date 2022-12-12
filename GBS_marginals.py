from typing import List, Tuple
import numpy as np
import copy 
from thewalrus import tor
from utils import get_click_indices, get_binary_basis

class Marginal:

    def Ch(self, r: float) -> np.ndarray:
        """Returns Ch submatrix of the squeezing vector."""
        return np.array([[np.cosh(r), 0], [0,np.cosh(r)]])

    def Sh(self, r: float) -> np.ndarray:
        """Returns Sh submatrix of the squeezing vector."""
        return np.array([[np.sinh(r), 0], [0,np.sinh(r)]])

    def get_S(self, r_k: np.ndarray) -> np.ndarray:
        """Returns the squeezing matrix for the two-mode squeezing parameters
        r_k (following the construction of Google's paper)."""
        Ns = len(r_k)*2
        S_ch = np.zeros((Ns,Ns))
        S_sh = np.zeros((Ns,Ns))
        for count,value in enumerate(r_k):
            ch_mat = self.Ch(value)
            sh_mat = self.Sh(value)
            i = 2 * count
            S_ch[i:i+2, i:i+2] = ch_mat
            S_sh[i:i+2, i:i+2] = sh_mat
        S_firstcolumn = np.concatenate((S_ch,S_sh), axis = 0)
        S_secondcolumn = np.concatenate((S_sh,S_ch), axis = 0)
        S = np.concatenate((S_firstcolumn,S_secondcolumn),axis = 1)
        return S  

    def get_input_covariance_matrix(self, S: np.ndarray) -> np.ndarray:
        """Returns the matrix sigma_in needed to obtain the covariance matrix."""
        sigma_vac = np.identity(len(S))/2
        return S*sigma_vac*S.T
        
    def get_output_covariance_matrix(self, matrix: np.ndarray, r_k: np.ndarray) -> np.ndarray:
        """Returns the covariance matrix of a GBS experiment defined by an interferometer
        matrix (the transformation matrix T or an ideal unitary matrix) and the squeezing
        parameters r_k."""
        sigma_in = self.get_input_covariance_matrix(self.get_S(r_k))
        m_len, m_height = np.shape(matrix)
        m_transpose_len, m_transpose_height = np.shape(matrix.T)
        first = np.zeros((m_len * 2, m_height * 2))
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
        sigma_red = np.zeros((len(indices),len(indices)))
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
        new_cov_matrix = np.dot(cov_matrix, np.linalg.inv(cov_matrix + np.identity(dim)))
        factor = 1/(4*(np.pi**2)*np.sqrt(np.linalg.det(cov_matrix)))
        return factor*np.sqrt(np.linalg.det(2*np.pi*new_cov_matrix))

    def get_single_outcome_probability(self, bitstring: Tuple, sigma: np.ndarray) -> float:
        """Return probability of a single output detection pattern
        in a GBS experiment defined by the squeezing parameters and
        the transformation matrix (which determine sigma)."""
        set_S = get_click_indices(bitstring)
        if not set_S:
            return self.get_prob_all_zero_bitstring(sigma)
        else:
            sigma_inv_reduced = self.get_reduced_matrix(np.linalg.inv(sigma), set_S)
            O_s = np.identity(sigma_inv_reduced.shape[0]) - sigma_inv_reduced
            return tor(O_s) / np.sqrt(np.linalg.norm(sigma))

    def get_marginal_distribution(
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
        distr = [self.get_single_outcome_probability(string, reduced_sigma) for string in binary_basis]
        return np.array(distr)



    

    
    