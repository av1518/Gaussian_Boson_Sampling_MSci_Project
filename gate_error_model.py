#%%
import numpy as np
from scipy.stats import unitary_group

def is_unitary(matrix):
    conjugate_transpose = matrix.conjugate().transpose()
    return np.allclose(conjugate_transpose, np.linalg.inv(matrix))

def apply_random_deviations(matrix, stddev):
    '''Takes input unitary matrix and applies deviation to each element (real & imaginary
    based on a normal distribution with input stddev'''
    if not is_unitary(matrix):
        raise ValueError("The input matrix is not unitary.")
    deviation_array = np.random.normal(scale=stddev, size=matrix.shape).astype(complex)
    deviated_arr = matrix + deviation_array
    deviation_array_imag = np.random.normal(scale=stddev, size=matrix.shape).astype(complex)
    return deviated_arr + deviation_array_imag * 1j


arr = np.array([[1+2j, 3+4j], [5+6j, 7+8j]])
A = ideal_matrix = unitary_group.rvs(2) #doesn't give symmetric matrix( only unitary)
print(A)
deviated_arr = apply_random_deviations(A, 0.1)

print(deviated_arr)
