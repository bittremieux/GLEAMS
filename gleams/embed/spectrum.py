import functools
import math

import numba as nb
import numpy as np
from spectrum_utils.spectrum import MsmsSpectrum

from gleams.embed import config


@nb.njit
def _check_spectrum_valid(spectrum_mz: np.ndarray, min_peaks: int,
                          min_mz_range: float) -> bool:
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
    if spectrum.is_processed:
        return spectrum

    min_peaks = config.min_peaks
    min_mz_range = config.min_mz_range

    spectrum = spectrum.set_mz_range(mz_min, mz_max)
    if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
        spectrum.is_valid = False
        spectrum.is_processed = True
        return spectrum

    if config.remove_precursor_tolerance is not None:
        spectrum = spectrum.remove_precursor_peak(
            config.remove_precursor_tolerance, 'Da', 2)
        if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
            spectrum.is_valid = False
            spectrum.is_processed = True
            return spectrum

    spectrum = spectrum.filter_intensity(config.min_intensity,
                                         config.max_peaks_used)
    if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
        spectrum.is_valid = False
        spectrum.is_processed = True
        return spectrum

    scaling = config.scaling
    if scaling == 'sqrt':
        scaling = 'root'
    if scaling is not None:
        spectrum = spectrum.scale_intensity(scaling,
                                            max_rank=config.max_peaks_used)

    # Set a flag to indicate that the spectrum has been processed to avoid
    # reprocessing.
    spectrum.is_valid = True
    spectrum.is_processed = True

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
