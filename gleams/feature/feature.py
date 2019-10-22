import logging
import os
import pickle
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gleams import config
from gleams.feature import encoder, spectrum
from gleams.ms_io import ms_io


logger = logging.getLogger('gleams')


def _peaks_to_features(dataset: str, filename: str, metadata: pd.DataFrame,
                       enc: encoder.SpectrumEncoder)\
        -> Tuple[str, Optional[List[str]], Optional[List[np.ndarray]]]:
    """
    Convert the spectra with the given identifiers in the given file to a
    feature array.

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
    Tuple[str, Optional[List[str]], Optional[List[np.ndarray]]]
        A tuple of length 3 containing: the name of the file that has been
        converted, the identifiers (scan numbers) of the converted spectra, the
        converted spectra.
        If the given file does not contain any (valid) spectra to be converted,
        the final two elements of the tuple are None.
    """
    peak_filename = os.path.join(
        os.environ['GLEAMS_HOME'], 'data', 'peak', dataset, filename)
    if not os.path.isfile(peak_filename):
        logger.warning('Missing peak file %s, no features generated',
                       peak_filename)
        return filename, None, None
    logger.debug('Process file %s/%s', dataset, filename)
    file_scans, file_encodings = [], []
    metadata = metadata.set_index('scan')
    for spec in ms_io.get_spectra(peak_filename):
        scan = str(spec.identifier)
        if (scan in metadata.index and
                spectrum.preprocess(spec, config.fragment_mz_min,
                                    config.fragment_mz_max).is_valid):
            file_scans.append(scan)
            file_encodings.append(enc.encode(spec))

    return filename, file_scans, file_encodings


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
        Feature files will be stored in the `dataset` subdirectory of this root
        directory.
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
    if not os.path.isdir(os.path.join(feat_dir, 'dataset')):
        try:
            os.makedirs(os.path.join(feat_dir, 'dataset'))
        except OSError:
            pass
    dataset_total = len(metadata.index.unique('dataset'))
    for dataset_i, (dataset, metadata_dataset) in enumerate(
            metadata.groupby('dataset'), 1):
        # Group all encoded spectra per dataset.
        filename_encodings = os.path.join(
            config.feat_dir, 'dataset', f'{dataset}.npy')
        filename_index = os.path.join(
            config.feat_dir, 'dataset', f'{dataset}.parquet')
        if (not os.path.isfile(filename_encodings) or
                not os.path.isfile(filename_index)):
            logging.info('Process dataset %s [%3d/%3d]', dataset, dataset_i,
                         dataset_total)
            index_filenames, index_scans, encodings = [], [], []
            for filename, file_scans, file_encodings in\
                    joblib.Parallel(n_jobs=-1, backend='multiprocessing')(
                        joblib.delayed(_peaks_to_features)
                        (dataset, fn, md_fn, enc)
                        for fn, md_fn in metadata_dataset.groupby('filename')):
                if file_scans is not None and len(file_scans) > 0:
                    index_filenames.extend([filename] * len(file_scans))
                    index_scans.extend(file_scans)
                    encodings.extend(file_encodings)
            # Store the encoded spectra in a file per dataset.
            if len(index_filenames) > 0:
                np.save(filename_encodings, np.vstack(encodings))
                index_df = (pd.DataFrame({'filename': index_filenames,
                                          'scan': index_scans}))
                pq.write_table(pa.Table.from_pandas(index_df), filename_index)


def combine_features(metadata_filename: str, feat_dir: str) -> None:
    """
    Combine feature files for multiple datasets into a single feature file.

    Parameters
    ----------
    metadata_filename : str
        Features for all datasets included in the metadata will be combined.
    feat_dir : str
        Root feature directory.
    """
    datasets = pd.read_csv(metadata_filename, usecols=['dataset'],
                           squeeze=True).unique()
    logger.info('Combine features for metadata file %s containing %d datasets',
                metadata_filename, len(datasets))
    encodings, indexes = [], []
    for i, dataset in enumerate(datasets, 1):
        logger.debug('Append dataset %s [%3d/%3d]', dataset, i, len(datasets))
        dataset_encodings_filename = os.path.join(
            feat_dir, 'dataset', f'{dataset}.npy')
        dataset_index_filename = os.path.join(
            feat_dir, 'dataset', f'{dataset}.parquet')
        if (not os.path.isfile(dataset_encodings_filename) or
                not os.path.isfile(dataset_index_filename)):
            logger.warning('Missing features for dataset %s, skipping...',
                           dataset)
        else:
            encodings.append(np.load(dataset_encodings_filename))
            indexes.append(pq.read_table(dataset_index_filename))
    feat_filename = os.path.join(feat_dir, os.path.splitext(os.path.basename(
        metadata_filename))[0].replace('metadata', 'feature'))
    np.save(f'{feat_filename}.npy', np.vstack(encodings))
    pq.write_table(pa.concat_tables(indexes), f'{feat_filename}.parquet')
