import functools
import math

import numba as nb
import numpy as np
from spectrum_utils.spectrum import MsmsSpectrum


@nb.njit
def is_valid(spectrum_mz: np.ndarray, min_peaks: int, min_mz_range: float)\
        -> bool:
    """
    Check whether a spectrum is of high enough quality to be used.

    Parameters
    ----------
    spectrum_mz : np.ndarray
        M/z peaks of the sspectrum whose quality is checked.
    min_peaks : int
        Minimum number of peaks a spectrum has to contain.
    min_mz_range : float
        Minimum m/z range the spectrum's peaks need to cover.

    Returns
    -------
    bool
        True if the spectrum has enough peaks covering a wide enough mass
        range, False otherwise.
    """
    return (len(spectrum_mz) >= min_peaks and
            spectrum_mz[-1] - spectrum_mz[0] >= min_mz_range)


def preprocess(spectrum: MsmsSpectrum, mz_min, mz_max) -> MsmsSpectrum:
    # TODO: Extend preprocessing functionality.
    spectrum = spectrum.set_mz_range(mz_min, mz_max)

    return spectrum


@functools.lru_cache(maxsize=None)
def get_num_bins(min_mz: float, max_mz: float, bin_size: float) -> int:
    """
    Compute the number of bins over the given mass range for the given bin
    size.

    Parameters
    ----------
    min_mz : float
        The minimum m/z to include in the vector.
    max_mz : float
        The maximum m/z to include in the vector.
    bin_size : float
        The bin size in m/z used to divide the m/z range.

    Returns
    -------
    int
        The number of bins over the given mass range for the given bin size.
    """
    return math.ceil((max_mz - min_mz) / bin_size)


def to_vector(spectrum: MsmsSpectrum, min_mz: float, max_mz: float,
              bin_size: float, normalize: bool) -> np.ndarray:
    """
    Convert the given spectrum to a dense NumPy vector.

    Parameters
    ----------
    spectrum : MsmsSpectrum
        The spectrum to be converted to a vector.
    min_mz : float
        The minimum m/z to include in the vector.
    max_mz : float
        The maximum m/z to include in the vector.
    bin_size : float
        The bin size in m/z used to divide the m/z range.
    normalize : bool
        Normalize the vector to unit length or not.

    Returns
    -------
    np.ndarray
        The binned spectrum vector.
    """
    vector = np.zeros((get_num_bins(min_mz, max_mz, bin_size),), np.float32)

    for mz, intensity in zip(spectrum.mz, spectrum.intensity):
        bin_idx = int((mz - min_mz) / bin_size)
        vector[bin_idx] += intensity

    return vector / np.linalg.norm(vector) if normalize else vector
