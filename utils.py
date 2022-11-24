from typing import Tuple
from thewalrus import hafnian, tor
import numpy as np

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

def haf(arr: np.ndarray):
    """Returns the hafnian of a matrix."""
    return hafnian(arr)

def toron(arr: np.ndarray):
    """Returns the torotonian of a matrix."""
    return tor(arr)