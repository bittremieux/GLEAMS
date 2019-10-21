import logging
import os
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from spectrum_utils.spectrum import MsmsSpectrum

from gleams import config
from gleams.feature import encoder, spectrum
from gleams.ms_io import ms_io


logger = logging.getLogger('gleams')


def _peaks_to_features(dataset: str, filename: str, metadata: pd.DataFrame,
                       enc: encoder.SpectrumEncoder)\
        -> Iterator[Tuple[MsmsSpectrum, str]]:
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
    Iterator[Tuple[MsmsSpectrum, str]]
        Tuples of encoded spectra and their spectrum identifier (scan number).
    """
    peak_filename = os.path.join(
        os.environ['GLEAMS_HOME'], 'data', 'peak', dataset, filename)
    if not os.path.isfile(peak_filename):
        logger.warning('Missing peak file %s, no features generated',
                       peak_filename)
        return
    for spec in ms_io.get_spectra(peak_filename):
        scan = str(spec.identifier)
        if (scan in metadata.index and
                spectrum.preprocess(spec, config.fragment_mz_min,
                                    config.fragment_mz_max).is_valid):
            yield enc.encode(spec), scan


def convert_peaks_to_features(metadata_filename: str, feat_dir: str)\
        -> None:
    """
    Convert all peak files listed in the given metadata file to features.

    Encoded spectra will be stored as NumPy binary files for each dataset in
    the metadata. A corresponding index file for each dataset containing the
    peak filenames, spectrum identifiers, and indexes in the NumPy binary file
    will be stored as Parquet files.

    If both the NumPy binary file and the Parquet index file already exist, the
    corresponding dataset will _not_ be processed again.

    Parameters
    ----------
    metadata_filename : str
        The metadata file name.
    feat_dir : str
        The directory in which the feature files will be stored.
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

    logger.info('Convert peak files for metadata file %s', metadata_filename)
    if not os.path.isdir(feat_dir):
        try:
            os.makedirs(feat_dir)
        except OSError:
            pass
    dataset_total = len(metadata.index.unique('dataset'))
    for dataset_i, (dataset, metadata_dataset) in enumerate(
            metadata.groupby('dataset'), 1):
        # Group all encoded spectra per dataset.
        filename_encodings = os.path.join(config.feat_dir, f'{dataset}.npy')
        filename_index = os.path.join(config.feat_dir, f'{dataset}.parquet')
        if (not os.path.isfile(filename_encodings) or
                not os.path.isfile(filename_index)):
            logging.info('Process dataset %s [%3d/%3d]', dataset, dataset_i,
                         dataset_total)
            index_is, index_filenames, index_scans, encodings = [], [], [], []
            for filename, metadata_filename in metadata_dataset.groupby(
                    'filename'):
                logger.debug('Process file %s/%s', dataset, filename)
                metadata_filename = metadata_filename.set_index('scan')
                for i, (spec_enc, scan) in enumerate(_peaks_to_features(
                        dataset, filename, metadata_filename, enc)):
                    encodings.append(spec_enc)
                    index_filenames.append(filename)
                    index_scans.append(scan)
                    index_is.append(i)
            # Store the encoded spectra in a file per dataset.
            np.save(filename_encodings, np.vstack(encodings))
            dataset_df = (pd.DataFrame({'filename': index_filenames,
                                        'scan': index_scans,
                                        'index': index_is})
                          .set_index(['filename', 'scan']))
            pq.write_table(pa.Table.from_pandas(
                dataset_df, preserve_index=True), filename_index)
