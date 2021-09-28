import functools
import math
from typing import Optional

import numba as nb
import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg
from spectrum_utils.spectrum import MsmsSpectrum


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


def preprocess(spectrum: MsmsSpectrum,
               mz_min: float,
               mz_max: float,
               min_peaks: int,
               min_mz_range: float,
               remove_precursor_tolerance: Optional[float],
               min_intensity: float,
               max_peaks_used: int,
               scaling: Optional[str]) -> MsmsSpectrum:
    """
    Preprocess the given spectrum.

    Parameters
    ----------
    spectrum:  MsmsSpectrum
        The spectrum to be preprocessed.
    mz_min : float
        The minimum m/z value to be included.
    mz_max : float
        The maximum m/z value to be included.
    min_peaks : int
        Minimum number of peaks the spectrum needs to have to be considered
        valid.
    min_mz_range : float
        Minimum m/z range to be covered for the spectrum to be considered
        valid.
    remove_precursor_tolerance : Optional[float]
        Remove peaks within the given m/z of the precursor peak.
    min_intensity : float
        Discard peaks below the given minimum intensity.
    max_peaks_used : int
        Retain only the given number of most intense peaks.
    scaling : Optional[str]
        Perform optional intensity scaling.

    Returns
    -------
    np.ndarray
        The normalized peak intensities.
    """
    if spectrum.is_processed:
        return spectrum

    spectrum = spectrum.set_mz_range(mz_min, mz_max)
    if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
        spectrum.is_valid = False
        spectrum.is_processed = True
        return spectrum

    if remove_precursor_tolerance is not None:
        spectrum = spectrum.remove_precursor_peak(
            remove_precursor_tolerance, 'Da')
        if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
            spectrum.is_valid = False
            spectrum.is_processed = True
            return spectrum

    spectrum = spectrum.filter_intensity(min_intensity, max_peaks_used)
    if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
        spectrum.is_valid = False
        spectrum.is_processed = True
        return spectrum

    scaling = scaling
    if scaling == 'sqrt':
        scaling = 'root'
    if scaling is not None:
        spectrum = spectrum.scale_intensity(scaling, max_rank=max_peaks_used)

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


def to_vector(spectrum_mz: np.ndarray, spectrum_intensity: np.ndarray,
              min_mz: float, bin_size: float, num_bins: int)\
        -> ss.csr_matrix:
    """
    Convert the given spectrum to a binned sparse SciPy vector.

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
    ss.csr_matrix
        The binned spectrum vector.
    """
    bins = ((spectrum_mz - min_mz) / bin_size).astype(np.int32)
    vector = ss.csr_matrix(
        (spectrum_intensity, (np.repeat(0, len(spectrum_intensity)), bins)),
        shape=(1, num_bins), dtype=np.float32)
    return vector / scipy.sparse.linalg.norm(vector)


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
    # Find the matching peaks between both spectra.
    peak_match_scores, peak_match_idx = [], []
    peak_other_i = 0
    for peak_i, (peak_mz, peak_intensity) in enumerate(zip(mz, intensity)):
        # Advance while there is an excessive mass difference.
        while (peak_other_i < len(mz_other) - 1 and
               peak_mz - fragment_mz_tol > mz_other[peak_other_i]):
            peak_other_i += 1
        # Match the peaks within the fragment mass window if possible.
        peak_other_window_i = peak_other_i
        while (peak_other_window_i < len(mz_other) and
               abs(peak_mz - (mz_other[peak_other_window_i]))
               <= fragment_mz_tol):
            peak_match_scores.append(
                peak_intensity * intensity_other[peak_other_window_i])
            peak_match_idx.append((peak_i, peak_other_window_i))
            peak_other_window_i += 1

    score = 0
    if len(peak_match_scores) > 0:
        # Use the most prominent peak matches to compute the score (sort in
        # descending order).
        peak_match_scores_arr = np.asarray(peak_match_scores)
        peak_match_order = np.argsort(peak_match_scores_arr)[::-1]
        peak_match_scores_arr = peak_match_scores_arr[peak_match_order]
        peak_match_idx_arr = np.asarray(peak_match_idx)[peak_match_order]
        peaks_used, peaks_used_other = set(), set()
        for peak_match_score, peak_i, peak_other_i in zip(
                peak_match_scores_arr, peak_match_idx_arr[:, 0],
                peak_match_idx_arr[:, 1]):
            if (peak_i not in peaks_used and
                    peak_other_i not in peaks_used_other):
                score += peak_match_score
                # Make sure these peaks are not used anymore.
                peaks_used.add(peak_i)
                peaks_used_other.add(peak_other_i)

    return score
