from typing import List, Tuple
import numpy as np
import copy 
from thewalrus import tor
from utils import get_click_indices, get_binary_basis


class Marginal:

    def Ch(self, r: float) -> np.ndarray:
        """Returns Ch submatrix of the squeezing matrix."""
        return np.array([[np.cosh(r), 0], [0,np.cosh(r)]])

    def Sh(self, r: float) -> np.ndarray:
        """Returns Sh submatrix of the squeezing matrix."""
        return np.array([[np.sinh(r), 0], [0,np.sinh(r)]])

    def get_S(self, r_k: np.ndarray) -> np.ndarray:
        """Returns the squeezing matrix."""
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
        """Returns the covariance matrix of a GBS experiment defined by a matrix (the
        transformation matrix T or the ideal unitary matrix of the interferometer) and
        the squeezing parameters r_k."""
        sigma_in = self.get_input_covariance_matrix(self.get_S(r_k))
        m_len, m_height = np.shape(matrix)
        m_transpose_len, m_transpose_height = np.shape(matrix.T)
        first = np.zeros((m_len * 2, m_height * 2))
        second = first.T
        first[0:m_len, 0:m_height] = matrix
        first[m_len:, m_height:] = matrix.conjugate()
        second[0:m_transpose_len, 0:m_transpose_height] = matrix.T.conjugate()
        second[m_transpose_len:, m_transpose_height:] = matrix.T
        if np.allclose(matrix.dot(matrix.conj().T), np.identity(matrix.shape[0]), atol=1e-12):
            return np.dot(first, np.dot(sigma_in, second))
        else:
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

    def get_single_outcome_probability(self, bitstring: Tuple, sigma: np.ndarray) -> float:
        """Return probability of a single output detection pattern
        in a GBS experiment defined by the squeezing parameters and
        the transformation matrix (which determine sigma)."""
        set_S = get_click_indices(bitstring)
        if not set_S:
            O_s = np.random.rand(sigma.shape[0], sigma.shape[0])
            #O_s = np.identity(sigma.shape[0])
        else:
            sigma_inv_reduced = self.get_reduced_matrix(np.linalg.inv(sigma), set_S)
            O_s = np.identity(sigma_inv_reduced.shape[0]) - sigma_inv_reduced
        return tor(O_s) / np.sqrt(np.linalg.norm(sigma))

    def get_marginal_distribution(self, mode_indices: List, sigma: np.ndarray) -> np.ndarray:
        """Returns marginal distribution of the specified modes."""
        k_order = len(mode_indices)
        binary_basis = get_binary_basis(k_order)
        reduced_sigma = self.get_reduced_matrix(sigma, mode_indices)
        distr = [self.get_single_outcome_probability(string, reduced_sigma) for string in binary_basis]
        return np.array(distr)



    

    
    