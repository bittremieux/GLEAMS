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


@nb.njit
def _norm_intensity(spectrum_intensity: np.ndarray) -> np.ndarray:
    """
    Normalize spectrum peak intensities.
    Parameters
    ----------
    spectrum_intensity : np.ndarray
        The spectrum peak intensities to be normalized.
    Returns
    -------
    np.ndarray
        The normalized peak intensities.
    """
    return spectrum_intensity / np.linalg.norm(spectrum_intensity)


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

    spectrum.intensity = _norm_intensity(spectrum.intensity)

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


@nb.njit
def to_vector(spectrum_mz: np.ndarray, spectrum_intensity: np.ndarray,
              min_mz: float, bin_size: float, num_bins: int)\
        -> np.ndarray:
    """
    Convert the given spectrum to a dense NumPy vector.

    Parameters
    ----------
    spectrum_mz : np.ndarray
        The peak m/z values of the spectrum to be converted to a vector.
    spectrum_intensity : np.ndarray
        The peak intensities of the spectrum to be converted to a vector.
    min_mz : float
        The minimum m/z to include in the vector.
    bin_size : float
        The bin size in m/z used to divide the m/z range.
    num_bins : int
        The number of elements of which the vector consists.

    Returns
    -------
    np.ndarray
        The binned spectrum vector.
    """
    vector = np.zeros((num_bins,), np.float32)

    for mz, intensity in zip(spectrum_mz, spectrum_intensity):
        bin_idx = int((mz - min_mz) / bin_size)
        vector[bin_idx] += intensity

    return vector / np.linalg.norm(vector)


@nb.njit
def dot(mz: np.ndarray, intensity: np.ndarray, mz_other: np.ndarray,
        intensity_other: np.ndarray, fragment_mz_tol: float) -> float:
    """
    Compute the dot product between two spectra.

    Note: Spectrum intensities should be normalized prior to computing the dot
    product.

    Parameters
    ----------
    mz : np.ndarray
        The first spectrum's m/z values.
    intensity : np.ndarray
        The first spectrum's intensity values.
    mz_other : np.ndarray
        The second spectrum's m/z values.
    intensity_other : np.ndarray
        The second spectrum's intensity values.
    fragment_mz_tol : float
        The fragment m/z tolerance used to match peaks in both spectra with
        each other.

    Returns
    -------
    float
        The dot product between both spectra.
    """
    fragment_i, fragment_other_i, score = 0, 0, 0.
    for fragment_i in range(len(mz)):
        while (mz_other[fragment_other_i] < mz[fragment_i] - fragment_mz_tol
               and fragment_other_i < len(mz_other)):
            fragment_other_i += 1
        if abs(mz[fragment_i] - mz_other[fragment_other_i]) <= fragment_mz_tol:
            score += intensity[fragment_i] * intensity_other[fragment_other_i]
            fragment_other_i += 1
    return score
