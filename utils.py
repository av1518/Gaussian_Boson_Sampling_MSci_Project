from typing import List, Tuple
import numpy as np
import math

def int_to_bitstring(integer: int) -> Tuple[int, ...]:
    """Converts an integer into a bitstring."""
    str = bin(integer)[2:]
    binary_number = tuple([int(bit) for bit in str])
    return binary_number

def int_to_padded_bitstring(integer: int, length: int) -> Tuple[int, ...]:
    """Converts an integer into a padded bitstring."""
    str = bin(integer)[2:].zfill(length)
    binary_number = tuple([int(bit) for bit in str])
    return binary_number
    
def bitstring_to_int(bitstring: Tuple[int, ...]) -> int:
    """Converts a bitstring into an integer."""
    suma = 0
    for i, bit in enumerate(bitstring):
        suma += bit*2**(len(bitstring) - i - 1)
    return int(suma)

def get_click_indices(bitstring: Tuple) -> List:
    """Returns indices of the 1s in a bitstring."""
    return [i for i, bit in enumerate(bitstring) if bit != 0]

def get_binary_basis(bit_number: int) -> List:
    """Returns complete binary basis for a given number of bits."""
    return [int_to_padded_bitstring(x, bit_number) for x in range(2**bit_number)]

def convert_to_clicks(outcomes: List) -> List:
    """Converts list of photon number patterns (tuples) into
    click patterns i.e. only distinguish between detection or
    no detection."""
    mutable_outcomes = [list(y) for y in outcomes]
    for outcome in mutable_outcomes:
        for i, x in enumerate(outcome):
            if x > 0:
                outcome[i] = 1
    return [tuple(y) for y in mutable_outcomes]

def total_variation_distance(distr1: np.ndarray, distr2: np.ndarray) -> float:
    """Returns total variation distance of two distributions."""
    return 0.5*np.sum(np.abs(distr1 - distr2))

def complex_to_polar(D_comp):
    '''
    D_comp == list of complex numbers (with r = 1)
    Returns the polar angle
    '''
    angles = []
    for i in D_comp:
        _,phi = math.polar(i)
        if phi < 0 :
            phi = 2*np.pi + phi
        angles.append(phi )
    return angles

def apply_random_deviation(input_matrix, standard_deviation):
    '''Takes input matrix (nested lists) and applied a normal 
    distribution deviation on the 3rd and 4th elements
    (use in gate error model)'''
    output_list = []
    for sublist in input_matrix:
        deviation_3 = np.random.normal(loc=0, scale=standard_deviation)
        deviation_4 = np.random.normal(loc=0, scale=standard_deviation)
        sublist_new = sublist.copy()
        sublist_new[2] += deviation_3
        sublist_new[3] += deviation_4
        output_list.append(sublist_new)
    return output_list

def kl_divergence(distr1: np.ndarray, distr2: np.ndarray):
    return np.sum(np.where(distr1 != 0, distr1 * np.log(distr1 / distr2), 0))