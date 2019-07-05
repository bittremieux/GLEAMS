import os
import re

import numpy as np


regex_non_alpha = re.compile('[^A-Za-z]+')

averagine_peak_separation_da = 1.0005079

hydrogen_mass = 1.00794


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
    # Divide the value range into equal intervals
    # and find the value's integer index.
    num_bins = 2 ** num_bits
    bin_size = (max_value - min_value) / num_bins
    bin_index = int((value - min_value) / bin_size)
    # Clip to min/max.
    bin_index = max(0, min(num_bins - 1, bin_index))
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


def get_data_path(filename: str) -> str:
    """
    Get the absolute path of a file in the expected data directory.

    Parameters
    ----------
    filename : str
        The name of the file in the data directory.

    Returns
    -------
    str
        The absolute path of the given file in the data directory.
    """
    return os.path.abspath(os.path.join(os.path.split(__file__)[0],
                                        '../..', 'data', filename))
