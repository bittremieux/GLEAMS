import logging
import os

import numpy as np
import pandas as pd

from gleams import config
from gleams.feature import encoder, spectrum
from gleams.ms_io import ms_io


logger = logging.getLogger('gleams')


def _peaks_to_features(dataset: str, filename: str,
                       identifiers_to_include: set,
                       enc: encoder.SpectrumEncoder) -> None:
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
    identifiers_to_include : set
        The identifiers of the spectra in the given file that will be converted
        to features.
    enc : encoder.SpectrumEncoder
        The SpectrumEncoder used to convert spectra to features.
    """
    peak_filename = os.path.join(
        os.environ['GLEAMS_HOME'], 'data', 'peak', dataset, filename)
    feat_filename = os.path.join(
        os.environ['GLEAMS_HOME'], 'data', 'feature', dataset,
        f'{os.path.splitext(filename)[0]}.npz')
    if not os.path.isfile(feat_filename):
        os.makedirs(os.path.dirname(feat_filename))
        features = [enc.encode(spec)
                    for spec in ms_io.get_spectra(peak_filename)
                    if spec.identifier in identifiers_to_include and
                    spectrum.preprocess(spec, config.fragment_mz_min,
                                        config.fragment_mz_max).is_valid]
        logger.debug('Save features to file %s', feat_filename)
        np.savez_compressed(feat_filename, np.vstack(features))


def convert_peaks_to_features(metadata_filename: str):
    """
    Convert all peak files listed in the given metadata file to features.

    Features will be stored as data/feature/{dataset}/{filename}.npz files
    corresponding to the original peak files.
    Existing feature files will _not_ be converted again.

    Parameters
    ----------
    metadata_filename : str
        The metadata file name. Should have a .csv extension.
    """
    metadata = pd.read_csv(metadata_filename, index_col=['dataset',
                                                         'filename'])

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

    logger.info('Convert peak files for metadata file %s to features',
                metadata_filename)
    for (dataset, filename), metadata_filename in metadata.groupby(
            level=['dataset', 'filename']):
        logger.debug('Convert peak file %s/%s to features', dataset, filename)
        _peaks_to_features(dataset, filename, set(metadata_filename['scan']),
                           enc)


def merge_features(metadata_filename: str):
    """
    Merge all feature files for the given metadata file into a single large
    feature file.

    If this file already exists it will _not_ be recreated.

    Parameters
    ----------
    metadata_filename : str
        The metadata file name.
    """
    feat_filename = os.path.join(
        os.environ['GLEAMS_HOME'], 'data', 'feature',
        (os.path.splitext(os.path.basename(metadata_filename))[0]
         .replace('metadata_', 'feature_') + '.npz'))
    if not os.path.isfile(feat_filename):
        logger.info('Merge feature files for metadata file %s',
                    metadata_filename)
        metadata = pd.read_csv(metadata_filename, index_col=['dataset',
                                                             'filename'])
        datasets_filenames = metadata.index.unique()
        features = [
            np.load(os.path.join(
                os.environ['GLEAMS_HOME'], 'data', 'feature', dataset,
                f'{os.path.splitext(filename)[0]}.npz'))['arr_0']
            for dataset, filename in zip(
                datasets_filenames.get_level_values('dataset'),
                datasets_filenames.get_level_values('filename'))]
        logger.debug('Save merged features to file %s', feat_filename)
        np.savez_compressed(feat_filename, np.vstack(features))
