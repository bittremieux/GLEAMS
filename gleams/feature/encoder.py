import abc
import itertools
import logging
import random
from typing import List

import numba as nb
import numpy as np
from spectrum_utils.spectrum import MsmsSpectrum

from gleams.feature import spectrum
from gleams.ms_io import ms_io


logger = logging.getLogger('gleams')


class SpectrumEncoder(metaclass=abc.ABCMeta):
    """
    Abstract superclass for spectrum encoders.
    """

    feature_names = None

    def __init__(self):
        pass

    @abc.abstractmethod
    def encode(self, spec: MsmsSpectrum) -> np.ndarray:
        """
        Encode the given spectrum.

        Parameters
        ----------
        spec : MsmsSpectrum
            The spectrum to be encoded.

        Returns
        -------
        np.ndarray
            Encoded spectrum features.
        """
        pass


class PrecursorEncoder(SpectrumEncoder):
    """
    Represents a spectrum as precursor features: gray encoding of the precursor
    m/z, gray encoding of the precursor neutral mass, and a one-hot encoding of
    the precursor charge.
    """

    def __init__(self, num_bits_mz: int, mz_min: float, mz_max: float,
                 num_bits_mass: int, mass_min: float, mass_max: float,
                 charge_max: int):
        """
        Instantiate a PrecursorEncoder.

        Parameters
        ----------
        num_bits_mz : int
            The number of bits to use to encode the precursor m/z.
        mz_min : float
            The minimum value between which to scale the precursor m/z.
        mz_max : float
            The maximum value between which to scale the precursor m/z.
        num_bits_mass : int
            The number of bits to use to encode the precursor neutral mass.
        mass_min : float
            The minimum value between which to scale the precursor neutral
            mass.
        mass_max : float
            The maximum value between which to scale the precursor neutral
            mass.
        charge_max : int
            The number of bits to use to encode the precursor charge. Higher
            charges will be clipped to this value.
        """
        super().__init__()

        self.num_bits_mz = num_bits_mz
        self.mz_min = mz_min
        self.mz_max = mz_max
        self.num_bits_mass = num_bits_mass
        self.mass_min = mass_min
        self.mass_max = mass_max
        self.charge_max = charge_max

        self.feature_names = [
            *[f'precursor_mz_{i}' for i in range(self.num_bits_mz)],
            *[f'precursor_mass_{i}' for i in range(self.num_bits_mass)],
            *[f'precursor_charge_{i}' for i in range(self.charge_max)]]

    def encode(self, spec: MsmsSpectrum) -> np.ndarray:
        """
        Encode the precursor of the given spectrum.

        Parameters
        ----------
        spec : MsmsSpectrum
            The spectrum to be encoded.

        Returns
        -------
        np.ndarray
            Spectrum precursor features consisting of: a gray encoding of the
            precursor m/z, a gray encoding of the precursor neutral mass, and
            a one-hot encoding of the precursor charge.
        """
        gray_code_mz = binary_encode(
            spec.precursor_mz, self.mz_min, self.mz_max, self.num_bits_mz)
        precursor_mass = neutral_mass_from_mz_charge(
            spec.precursor_mz, spec.precursor_charge)
        gray_code_mass = binary_encode(
            precursor_mass, self.mass_min, self.mass_max, self.num_bits_mass)
        one_hot_charge = np.zeros((self.charge_max,), np.float32)
        one_hot_charge[min(spec.precursor_charge, self.charge_max) - 1] = 1.
        return np.hstack([gray_code_mz, gray_code_mass, one_hot_charge])


class FragmentEncoder(SpectrumEncoder):
    """
    Represents a spectrum as a vector of fragment ions.
    """

    def __init__(self, min_mz: float, max_mz: float, bin_size: float):
        """
        Instantiate a FragmentEncoder.

        Parameters
        ----------
        min_mz : float
            The minimum m/z to use for spectrum vectorization.
        max_mz : float
            The maximum m/z to use for spectrum vectorization.
        bin_size : float
            The bin size in m/z used to divide the m/z range.
        """
        super().__init__()

        self.min_mz = min_mz
        self.max_mz = max_mz
        self.bin_size = bin_size
        self.num_bins = spectrum.get_num_bins(min_mz, max_mz, bin_size)

        self.feature_names = [f'fragment_bin_{i}'
                              for i in range(self.num_bins)]

    def encode(self, spec: MsmsSpectrum) -> np.ndarray:
        """
        Encode the fragments of the given spectrum.

        Parameters
        ----------
        spec : MsmsSpectrum
            The spectrum to be encoded.

        Returns
        -------
        np.ndarray
            Spectrum fragment features consisting a vector of binned fragments.
        """
        return spectrum.to_vector(spec.mz, spec.intensity, self.min_mz,
                                  self.bin_size, self.num_bins)


class ReferenceSpectraEncoder(SpectrumEncoder):
    """
    Represents a spectrum as similarity to a set of reference spectra.
    """

    def __init__(self, filename: str, min_mz: float, max_mz: float,
                 fragment_mz_tol: float, num_ref_spectra: int):
        """
        Instantiate a ReferenceSpectraEncoder by vectorizing the reference
        spectra in the given file.

        Parameters
        ----------
        filename : str
            The file from which to read the reference spectra.
        min_mz : float
            The minimum m/z to include in the vector.
        max_mz : float
            The maximum m/z to include in the vector.
        fragment_mz_tol : float
            The fragment m/z tolerance used to compute the spectrum dot
            product to the reference spectra.
        num_ref_spectra : int
            Maximum number of reference spectra to consider. An error is raised
            if this exceeds the number of available reference spectra.
        """
        super().__init__()

        self.fragment_mz_tol = fragment_mz_tol

        logger.debug('Read the reference spectra')
        ref_spectra = list(ms_io.get_spectra(filename))
        if len(ref_spectra) < num_ref_spectra:
            raise ValueError(f'Insufficient number of reference spectra '
                             f'({len(ref_spectra)} available, '
                             f'{num_ref_spectra} required)')
        elif len(ref_spectra) > num_ref_spectra:
            logger.debug('Select %d reference spectra (was %d)',
                         len(ref_spectra), num_ref_spectra)
            ref_spectra = random.sample(ref_spectra, num_ref_spectra)
        logger.debug('Vectorize the reference spectra')
        self.ref_spectra = [spec for spec in ref_spectra
                            if (spectrum.preprocess(spec, min_mz, max_mz)
                                .is_valid)]

        self.feature_names = [f'ref_{i}' for i in range(len(ref_spectra))]

    def encode(self, spec: MsmsSpectrum) -> np.ndarray:
        """
        Encode the given spectrum by its similarity with a set of reference
        spectra.

        Parameters
        ----------
        spec : MsmsSpectrum
            The spectrum to be encoded.

        Returns
        -------
        np.ndarray
            Reference spectrum features consisting of the spectrum's dot
            product similarity to a set of reference spectra.
        """
        return np.asarray([spectrum.dot(ref.mz, ref.intensity, spec.mz,
                                        spec.intensity, self.fragment_mz_tol)
                           for ref in self.ref_spectra])


class MultipleEncoder(SpectrumEncoder):
    """
    Combines multiple child encoders.
    """

    def __init__(self, encoders: List[SpectrumEncoder]):
        """
        Instantiate a MultipleEncoder with the given child encoders.

        Parameters
        ----------
        encoders : List[SpectrumEncoder]
            The child encoders to do the actual spectrum encoding.
        """
        super().__init__()

        self.encoders = encoders

        self.feature_names = list(itertools.chain.from_iterable(
            [enc.feature_names for enc in self.encoders]))

    def encode(self, spec: MsmsSpectrum) -> np.ndarray:
        """
        Encode the given spectrum using the child encoders.

        Parameters
        ----------
        spec : MsmsSpectrum
            The spectrum to be encoded.

        Returns
        -------
        np.ndarray
            Concatenated spectrum features produced by all child encoders.
        """
        return np.hstack([enc.encode(spec) for enc in self.encoders])


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
    hydrogen_mass = 1.00794
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
