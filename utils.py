from typing import Tuple

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
