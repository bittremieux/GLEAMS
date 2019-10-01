import logging
import os
from typing import Iterator, Tuple

import h5py
import numpy as np
import pandas as pd
from spectrum_utils.spectrum import MsmsSpectrum

from gleams import config
from gleams.feature import encoder, spectrum
from gleams.ms_io import ms_io


logger = logging.getLogger('gleams')


def _peaks_to_features(dataset: str, filename: str, metadata: pd.DataFrame,
                       enc: encoder.SpectrumEncoder)\
        -> Iterator[Tuple[MsmsSpectrum, np.ndarray]]:
    """
    Convert the spectra with the given identifiers in the given file to a
    feature array.

    If the feature file already exist it will _not_ be recreated.

    Parameters
    ----------
    dataset : str
        The peak file's dataset.
    filename : str
        The peak file name.
    metadata : pd.DataFrame
        DataFrame containing metadata for the PSMs in the peak file to be
        processed.
    enc : encoder.SpectrumEncoder
        The SpectrumEncoder used to convert spectra to features.

    Returns
    -------
    Iterator[Tuple[MsmsSpectrum, np.ndarray]]
        Tuples of spectra and their encoded feature array.
    """
    peak_filename = os.path.join(
        os.environ['GLEAMS_HOME'], 'data', 'peak', dataset, filename)
    if not os.path.isfile(peak_filename):
        logger.warning('Missing peak file %s, no features generated',
                       peak_filename)
        return
    logger.debug('Convert peak file %s/%s to features', dataset, filename)
    for spec in ms_io.get_spectra(peak_filename):
        if (str(spec.identifier) in metadata.index and
                spectrum.preprocess(spec, config.fragment_mz_min,
                                    config.fragment_mz_max).is_valid):
            spec.peptide = metadata.at[spec.identifier, 'sequence']
            yield spec, enc.encode(spec)


def convert_peaks_to_features(metadata_filename: str, feat_filename: str)\
        -> None:
    """
    Convert all peak files listed in the given metadata file to features.

    Features will be stored as in the given HDF5 file.
    Existing features will _not_ be converted again.

    Parameters
    ----------
    metadata_filename : str
        The metadata file name.
    feat_filename : str
        The HDF5 feature file name.
    """
    metadata = pd.read_csv(metadata_filename,
                           index_col=['dataset', 'filename'],
                           dtype={'scan': str})

    enc = encoder.MultipleEncoder([
        encoder.PrecursorEncoder(
            config.num_bits_precursor_mz, config.precursor_mz_min,
            config.precursor_mz_max, config.num_bits_precursor_mass,
            config.precursor_mass_min, config.precursor_mass_max,
            config.precursor_charge_max),
        encoder.FragmentEncoder(
            config.fragment_mz_min, config.fragment_mz_max, config.bin_size),
        encoder.ReferenceSpectraEncoder(
            config.ref_spectra_filename, config.fragment_mz_min,
            config.fragment_mz_max, config.fragment_mz_tol,
            config.num_ref_spectra)
    ])

    feat_dir = os.path.dirname(feat_filename)
    logger.info('Convert peak files for metadata file %s to features file %s',
                metadata_filename, feat_filename)
    if not os.path.isdir(feat_dir):
        try:
            os.makedirs(feat_dir)
        except OSError:
            pass
    with h5py.File(feat_filename, 'a') as f_feat:
        for (dataset, filename), metadata_filename in metadata.groupby(
                level=['dataset', 'filename']):
            if f'{dataset}{filename}' not in f_feat:
                metadata_filename = metadata_filename.set_index('scan')
                for spec, spec_enc in _peaks_to_features(
                        dataset, filename, metadata_filename, enc):
                    spec_hdf5 = f_feat.create_dataset(
                        f'{dataset}/{filename}/{spec.identifier}',
                        data=spec_enc, compression='lzf')
                    spec_hdf5.attrs['sequence'] = spec.peptide
                    spec_hdf5.attrs['charge'] = spec.precursor_charge
                    spec_hdf5.attrs['mz'] = spec.precursor_mz
