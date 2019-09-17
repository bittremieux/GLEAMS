import abc
import functools
import itertools
import logging
import random
from typing import Dict, Generator, List, Tuple

import numpy as np
import pandas as pd
from spectrum_utils.spectrum import MsmsSpectrum

from gleams import config
from gleams.embed import spectrum, theoretical, utils
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
        gray_code_mz = utils.binary_encode(
            spec.precursor_mz, self.mz_min, self.mz_max, self.num_bits_mz)
        precursor_mass = utils.neutral_mass_from_mz_charge(
            spec.precursor_mz, spec.precursor_charge)
        gray_code_mass = utils.binary_encode(
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


class PairGenerator:
    """
    Generate spectrum pairs.
    """

    def __init__(self):
        """
        Instantiate the PairGenerator.

        The precursor m/z difference for real–real spectrum pairs will be at
        most `config.pair_mz_tolerance`.

        Simulated spectra will be generated using MS2PIP using the
        `config.ms2pip_model` model.
        """
        self.spectra = None
        self.metadata = None
        self.mz_tolerance = config.pair_mz_tolerance
        self.spectrum_simulator = theoretical.SpectrumSimulator(
            config.ms2pip_model)

    def set_spectra(self, spectra: Dict[str, MsmsSpectrum],
                    metadata: pd.DataFrame) -> 'PairGenerator':
        """
        Specify the spectra for which spectrum pairs are generated.

        Parameters
        ----------
        spectra : Dict[str, MsmsSpectrum]
            A dictionary of the spectra for which spectrum pairs are generated.
            The dictionary keys are the identifiers of the spectra.
        metadata : pd.DataFrame
            Metadata associated with the given spectra. The metadata dataframe
            should have as index the spectrum identifiers, and have "sequence",
            "charge", and "mz" columns.

        Returns
        -------
        PairGenerator
        """
        self.spectra = spectra
        self.metadata = metadata
        return self

    def generate_pairs(self, real: bool = True, positive: bool = True)\
            -> Generator[Tuple[MsmsSpectrum, MsmsSpectrum], None, None]:
        """
        Generator producing spectrum pairs.

        Parameters
        ----------
        real : bool
            Flag indicating whether real or simulated spectrum pairs are
            generated.
        positive : bool
            Flag indicating whether positive or negative pairs are generated.

        Returns
        -------
        Generator[Tuple[MsmsSpectrum, MsmsSpectrum], None, None]
            A generator of spectrum–spectrum tuples.
        """
        if self.spectra is None or self.metadata is None:
            raise ValueError('Provide suitable spectra and metadata '
                             'information to generate spectra pairs')
        if real:
            logger.info('Compile real–real %s spectrum pairs for %d spectra',
                        'positive' if positive else 'negative',
                        len(self.spectra))
            pairs_seen = set()
            for spec in self.spectra.values():
                peptide = self.metadata.at[spec.identifier, 'sequence']
                mz_selector = self.metadata['mz'].between(
                    spec.precursor_mz - self.mz_tolerance,
                    spec.precursor_mz + self.mz_tolerance)
                sequence_selector = self.metadata['sequence'] == peptide
                if not positive:
                    sequence_selector = ~sequence_selector
                spectra_i = self.metadata[mz_selector & sequence_selector]
                if positive:
                    pair_generator = functools.partial(
                        itertools.combinations_with_replacement, r=2)
                else:
                    def pair_generator(i2s): yield from zip(
                        itertools.repeat(spec.identifier), i2s)
                for i1, i2 in pair_generator(spectra_i.index):
                    pair_key = tuple(sorted((i1, i2)))
                    if pair_key not in pairs_seen:
                        pairs_seen.add(pair_key)
                        yield self.spectra[i1], self.spectra[i2]
        else:
            logger.info('Compile real–simulated %s spectrum pairs for %d '
                        'spectra', 'positive' if positive else 'negative',
                        len(self.spectra))
            yield from zip(
                self.spectra.values(),
                self.spectrum_simulator.simulate(
                    self.metadata.loc[self.spectra.keys(), 'sequence'],
                    self.metadata.loc[self.spectra.keys(), 'charge'],
                    not positive))
