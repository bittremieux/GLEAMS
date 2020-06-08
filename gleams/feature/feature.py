import logging
import os
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as ss

from gleams import config
from gleams.feature import encoder, spectrum
from gleams.ms_io import ms_io


logger = logging.getLogger('gleams')


def _peaks_to_features(dataset: str, filename: str,
                       metadata: Optional[pd.DataFrame],
                       enc: encoder.SpectrumEncoder)\
        -> Tuple[str, Optional[pd.DataFrame], Optional[List[np.ndarray]]]:
    """
    Convert the spectra with the given identifiers in the given file to a
    feature array.

    Parameters
    ----------
    dataset : str
        The peak file's dataset.
    filename : str
        The peak file name.
    metadata : Optional[pd.DataFrame]
        DataFrame containing metadata for the PSMs in the peak file to be
        processed. If None, all spectra in the peak file are converted to
        features.
    enc : encoder.SpectrumEncoder
        The SpectrumEncoder used to convert spectra to features.

    Returns
    -------
    Tuple[str, Optional[pd.DataFrame], Optional[List[np.ndarray]]]
        A tuple of length 3 containing: the name of the file that has been
        converted, information about the converted spectra (scan number,
        precursor charge, and precursor m/z), the converted spectra.
        If the given file does not exist the final two elements of the tuple
        are None.
    """
    peak_filename = os.path.join(
        os.environ['GLEAMS_HOME'], 'data', 'peak', dataset, filename)
    if not os.path.isfile(peak_filename):
        logger.warning('Missing peak file %s, no features generated',
                       peak_filename)
        return filename, None, None
    logger.debug('Process file %s/%s', dataset, filename)
    file_scans, file_mz, file_charge, file_encodings = [], [], [], []
    if metadata is not None:
        metadata = metadata.reset_index(['dataset', 'filename'], drop=True)
    for spec in ms_io.get_spectra(peak_filename):
        if ((metadata is None or np.int64(spec.identifier) in metadata.index)
                and spectrum.preprocess(spec, config.fragment_mz_min,
                                        config.fragment_mz_max).is_valid):
            file_scans.append(spec.identifier)
            file_mz.append(spec.precursor_mz)
            file_charge.append(spec.precursor_charge)
            file_encodings.append(enc.encode(spec))
    scans = pd.DataFrame({'scan': file_scans, 'charge': file_charge,
                          'mz': file_mz})
    scans['scan'] = scans['scan'].astype(np.int64)
    return filename, scans, file_encodings


def convert_peaks_to_features(metadata_filename: str)\
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
        The metadata file name. Should be a Parquet file.
    """
    metadata = pd.read_parquet(metadata_filename)
    metadata = metadata.set_index(['dataset', 'filename', 'scan'])

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
    feat_dir = os.path.join(os.environ['GLEAMS_HOME'], 'data', 'feature',
                            'dataset')
    if not os.path.isdir(feat_dir):
        try:
            os.makedirs(os.path.join(feat_dir))
        except OSError:
            pass
    dataset_total = len(metadata.index.unique('dataset'))
    for dataset_i, (dataset, metadata_dataset) in enumerate(
            metadata.groupby('dataset', as_index=False, sort=False), 1):
        # Group all encoded spectra per dataset.
        feat_dir = os.path.join(os.environ['GLEAMS_HOME'], 'data', 'feature')
        filename_encodings = os.path.join(
            feat_dir, 'dataset', f'{dataset}.npz')
        filename_index = os.path.join(
            feat_dir, 'dataset', f'{dataset}.parquet')
        if (not os.path.isfile(filename_encodings) or
                not os.path.isfile(filename_index)):
            logging.info('Process dataset %s [%3d/%3d]', dataset, dataset_i,
                         dataset_total)
            metadata_index, encodings = [], []
            for filename, file_scans, file_encodings in\
                    joblib.Parallel(n_jobs=-1, backend='multiprocessing')(
                        joblib.delayed(_peaks_to_features)
                        (dataset, fn, md_fn, enc)
                        for fn, md_fn in metadata_dataset.groupby(
                            'filename', as_index=False, sort=False)):
                if file_scans is not None and len(file_scans) > 0:
                    metadata_index.extend([(dataset, filename, scan)
                                           for scan in file_scans['scan']])
                    encodings.extend(file_encodings)
            # Store the encoded spectra in a file per dataset.
            if len(metadata_index) > 0:
                ss.save_npz(filename_encodings, ss.vstack(encodings))
                metadata.loc[metadata_index].reset_index().to_parquet(
                    filename_index, index=False)


def combine_features(metadata_filename: str) -> None:
    """
    Combine feature files for multiple datasets into a single feature file.

    If the combined feature file already exists it will _not_ be recreated.

    Parameters
    ----------
    metadata_filename : str
        Features for all datasets included in the metadata will be combined.
        Should be a Parquet file.
    """
    feat_dir = os.path.join(os.environ['GLEAMS_HOME'], 'data', 'feature')
    feat_filename = os.path.join(feat_dir, os.path.splitext(
        os.path.basename(metadata_filename))[0].replace('metadata', 'feature'))
    if (os.path.isfile(f'{feat_filename}.npy') and
            os.path.isfile(f'{feat_filename}.parquet')):
        return
    datasets = pd.read_parquet(
        metadata_filename, columns=['dataset'])['dataset'].unique()
    logger.info('Combine features for metadata file %s containing %d datasets',
                metadata_filename, len(datasets))
    encodings, indexes = [], []
    for i, dataset in enumerate(datasets, 1):
        logger.debug('Append dataset %s [%3d/%3d]', dataset, i, len(datasets))
        dataset_encodings_filename = os.path.join(
            feat_dir, 'dataset', f'{dataset}.npz')
        dataset_index_filename = os.path.join(
            feat_dir, 'dataset', f'{dataset}.parquet')
        if (not os.path.isfile(dataset_encodings_filename) or
                not os.path.isfile(dataset_index_filename)):
            logger.warning('Missing features for dataset %s, skipping...',
                           dataset)
        else:
            encodings.append(ss.load_npz(dataset_encodings_filename))
            indexes.append(pq.read_table(dataset_index_filename))
    np.save(f'{feat_filename}.npz', ss.vstack(encodings))
    pq.write_table(pa.concat_tables(indexes), f'{feat_filename}.parquet')
