import os
import random
import re
import string
from typing import Iterator, Tuple

import numba as nb
import numpy as np


regex_non_alpha = re.compile('[^A-Za-z]+')
regex_modifications = re.compile('[A-Z]?[-+]?\d*\.\d+|\d+')

averagine_peak_separation_da = 1.0005079

hydrogen_mass = 1.00794


@nb.njit
def neutral_mass_from_mz_charge(mz: float, charge: int) -> float:
    """
    Calculate the neutral mass of an ion given its m/z and charge.

    Parameters
    ----------
    mz : float
        The ion's m/z.
    charge : int
        The ion's charge.

    Returns
    -------
    float
        The ion's neutral mass.
    """
    return (mz - hydrogen_mass) * charge


def _gray_code(value: int, num_bits: int) -> np.ndarray:
    """
    Return the Gray code for a given integer, given the number of bits to use
    for the encoding.

    Parameters
    ----------
    value : int
        The integer value to be converted to Gray code.
    num_bits : int
        The number of bits of the encoding. No checking is done to ensure a
        sufficient number of bits to store the encoding is specified.

    Returns
    -------
    np.ndarray
        An array of individual bit values as floats.
    """
    # Gray encoding: https://stackoverflow.com/a/38745459
    return np.asarray(list(f'{value ^ (value >> 1):0{num_bits}b}'), np.float32)


@nb.njit
def _get_bin_index(value: float, min_value: float, max_value: float,
                   num_bits: int) -> int:
    """
    Get the index of the given value between a minimum and maximum value given
    a specified number of bits.

    Parameters
    ----------
    value : float
        The value to be converted to a bin index.
    min_value : float
        The minimum possible value.
    max_value : float
        The maximum possible value.
    num_bits : int
        The number of bits of the encoding.

    Returns
    -------
    int
        The integer bin index of the value between the given minimum and
        maximum value using the specified number of bits.
    """
    # Divide the value range into equal intervals
    # and find the value's integer index.
    num_bins = 2 ** num_bits
    bin_size = (max_value - min_value) / num_bins
    bin_index = int((value - min_value) / bin_size)
    # Clip to min/max.
    bin_index = max(0, min(num_bins - 1, bin_index))
    return bin_index


def binary_encode(value: float, min_value: float, max_value: float,
                  num_bits: int) -> np.ndarray:
    """
    Return the Gray code for a given value, given the number of bits to use
    for the encoding. The given number of bits equally spans the range between
    the given minimum and maximum value.
    If the given value is not within the interval given by the minimum and
    maximum value it will be clipped to either extremum.

    Parameters
    ----------
    value : float
        The value to be converted to Gray code.
    min_value : float
        The minimum possible value.
    max_value : float
        The maximum possible value.
    num_bits : int
        The number of bits of the encoding.

    Returns
    -------
    np.ndarray
        An array of individual bit values as floats.
    """
    bin_index = _get_bin_index(value, min_value, max_value, num_bits)
    return _gray_code(bin_index, num_bits)


def normalize_peptide(peptide: str) -> str:
    """
    Normalize a peptide sequence. Non-alphabetic characters are removed and Is
    are replaced by Ls.

    Parameters
    ----------
    peptide : str
        The given peptide sequence string.

    Returns
    -------
    str
        The normalized peptide sequence, or None if the input was None.
    """
    return (regex_non_alpha.sub('', peptide).upper().replace('I', 'L')
            if peptide is not None else None)


def get_peptide_modifications(peptide: str)\
        -> Iterator[Tuple[float, str, int]]:
    """
    Extract modifications from a peptide sequence.

    Parameters
    ----------
    peptide : str
        The peptide sequence from which modifications are extracted.

    Returns
    -------
    Tuple[float, str, int]
        An iterator over tuples for each modification that is present in the
        peptide sequence consisting of the modification's mass difference,
        the amino acid on which the modification is active (or 'N-term' for
        N-terminal modifications), and the amino acid position of the
        modification (1-based, 0 for N-terminal modifications).
    """
    match = re.search(regex_modifications, peptide)
    while match is not None:
        if match[0][0].isalpha():
            # Internal modification.
            pos = match.start() + 1
            peptide = peptide[:pos] + peptide[match.end():]
            yield float(match[0][1:]), match[0][0], pos
        else:
            # N-terminal modification.
            peptide = peptide[match.end():]
            yield float(match[0]), 'N-term', 0
        match = re.search(regex_modifications, peptide)


def generate_random_string(length: int) -> str:
    """
    Generate a random alphabetic string.

    Parameters
    ----------
    length : int
        The length of the string.

    Returns
    -------
    A random string consisting of alphabetic characters.
    """
    return ''.join(random.choices(string.ascii_lowercase, k=length))
