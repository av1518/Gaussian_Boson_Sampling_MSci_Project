from typing import List, Tuple

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