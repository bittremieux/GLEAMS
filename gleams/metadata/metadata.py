import functools
import itertools
import logging
import os
import subprocess
import warnings
from typing import Iterator, List, Tuple

import joblib
import numba as nb
import numpy as np
import pandas as pd
from spectrum_utils import utils as suu


logger = logging.getLogger('gleams')


def convert_massivekb_metadata(massivekb_filename: str,
                               metadata_filename: str) -> None:
    """
    Convert the MassIVE-KB metadata file to a stripped down metadata file
    containing only the relevant information.
    The initial metadata file needs to be downloaded manually from MassIVE:
    MassIVE Knowledge Base > Human HCD Spectral Library
        > All Candidate library spectra > Download

    The new metadata file will contain PSM information the following columns:
    - dataset: The MassIVE dataset identifier.
    - filename: The file in which the PSM's spectrum is present.
    - scan: The PSM's scan number in its spectral data file.
    - sequence: The PSM's peptide sequence.
    - charge: The PSM's precursor charge.
    - mz: The PSM's precursor m/z.

    If the stripped down metadata file already exists it will _not_ be
    recreated.

    Parameters
    ----------
    massivekb_filename : str
        The MassIVE-KB metadata file name.
    metadata_filename : str
        The metadata file name.
    """
    if not os.path.isfile(metadata_filename):
        logger.info('Convert the MassIVE-KB metadata file')
        metadata = pd.read_csv(massivekb_filename, sep='\t', usecols=[
            'annotation', 'charge', 'filename', 'mz', 'scan'])
        metadata = metadata.rename(columns={'annotation': 'sequence'})
        dataset_filename = metadata['filename'].str.split('/').str
        metadata['dataset'] = dataset_filename[0]
        metadata['filename'] = dataset_filename[-1]
        metadata = metadata.sort_values(['dataset', 'filename', 'scan'])
        metadata = metadata[['dataset', 'filename', 'scan', 'sequence',
                             'charge', 'mz']]
        logger.debug('Save metadata file to %s', metadata_filename)
        metadata.to_parquet(metadata_filename, index=False)


def split_metadata_train_val_test(
        metadata_filename: str, val_ratio: float = None,
        test_ratio: float = None, rel_tol: float = None) -> None:
    """
    Split the metadata file in training/validation/test files.

    The split is based on dataset, with the ratio of the number of PSMs in each
    split approximating the given ratios.

    Parameters
    ----------
    metadata_filename : str
        The input metadata filename. Should be a Parquet file.
    val_ratio : float
        Proportion of the total number of PSMs that should approximately be in
        the validation set. If None, no validation split will be generated.
    test_ratio : float
        Proportion of the total number of PSMs that should approximately be in
        the test set. If None, no test split will be generated.
    rel_tol : float
        Proportion tolerance of the number of PSMs in the validation and test
        splits. Default: 0.1 * val_ratio.
    """
    filename_train = metadata_filename.replace('.parquet', '_train.parquet')
    filename_val = metadata_filename.replace('.parquet', '_val.parquet')
    filename_test = metadata_filename.replace('.parquet', '_test.parquet')
    if (os.path.isfile(filename_train) and
            (val_ratio is None or os.path.isfile(filename_val)) and
            (test_ratio is None or os.path.isfile(filename_test))):
        return
    metadata = pd.read_parquet(metadata_filename).set_index('dataset')
    abs_tol = int((rel_tol if rel_tol is not None else
                   (0.1 * (val_ratio if val_ratio is not None else 0)))
                  * len(metadata))
    num_val = int(val_ratio * len(metadata)) if val_ratio is not None else 0
    num_test = int(test_ratio * len(metadata)) if test_ratio is not None else 0
    # Add datasets to the validation/test splits until they contain a suitable
    # number of PSMs.
    perc_val = (val_ratio if val_ratio is not None else 0) * 100
    perc_test = (test_ratio if test_ratio is not None else 0) * 100
    perc_train = 100 - perc_val - perc_test
    logger.info('Split the metadata file into train (~%.f%%), validation '
                '(~%.f%%), and test (~%.f%%) sets', perc_train, perc_val,
                perc_test)
    datasets = (metadata.groupby('dataset', sort=False)['scan']
                .count().sample(frac=1))
    if num_val > 0:
        selected_val = _select_datasets(datasets, num_val, abs_tol)
        logger.debug('Save validation metadata file to %s', filename_val)
        metadata.loc[selected_val].to_parquet(filename_val, index=True)
        datasets = datasets.drop(selected_val)
    if num_test > 0:
        selected_test = _select_datasets(datasets, num_test, abs_tol)
        logger.debug('Save test metadata file to %s', filename_test)
        metadata.loc[selected_test].to_parquet(filename_test, index=True)
        datasets = datasets.drop(selected_test)
    logger.debug('Save train metadata file to %s', filename_train)
    metadata.loc[datasets.index].to_parquet(filename_train, index=True)


def _select_datasets(datasets: pd.Series, num_to_select: int, num_tol: int)\
        -> List[str]:
    """
    Select datasets with the specified number of PSMs until the requested
    number of PSMs is approximately reached.

    Parameters
    ----------
    datasets : pd.Series
        A Series with dataset identifiers as index and number of PSMs as
        values.
    num_to_select : int
        The number of PSMs that should approximately be selected.
    num_tol : int
        The amount of deviation that is allowed from `num_to_select`.

    Returns
    -------
    List[str]
        A list of dataset identifiers that are selected.
    """
    datasets_selected, num_selected = [], 0
    for dataset, dataset_num_psms in datasets.items():
        if (num_selected + dataset_num_psms) - num_to_select < num_tol:
            datasets_selected.append(dataset)
            num_selected += dataset_num_psms
        if abs(num_to_select - num_selected) <= num_tol:
            break
    return datasets_selected


def download_massive_file(massive_filename: str) -> None:
    """
    Download the given file from MassIVE.

    The file is downloaded using a `wget` subprocess.
    The file will be stored in the `data/peak/{dataset}/{filename}` directory.
    If the file already exists it will _not_ be downloaded again.

    Parameters
    ----------
    massive_filename : str
        The local MassIVE file link.
    """
    dataset = massive_filename.split('/', 1)[0]
    dataset_dir = os.path.join(os.environ['GLEAMS_HOME'], 'data', 'peak',
                               dataset)
    peak_filename = massive_filename.rsplit('/', 1)[-1]
    if not os.path.isfile(peak_filename):
        if not os.path.isdir(dataset_dir):
            try:
                os.makedirs(dataset_dir)
            except OSError:
                pass
        logger.debug('Download file %s/%s', dataset, peak_filename)
        url = f'ftp://massive.ucsd.edu/{massive_filename}'
        proc = subprocess.run(
            ['wget', '--no-verbose', '--timestamping', '--retry-connrefused',
             f'--directory-prefix={dataset_dir}', '--passive-ftp', url],
            capture_output=True, text=True)
        if proc.returncode != 0:
            logger.warning('Could not download file %s/%s: wget error %d: %s',
                           dataset, peak_filename, proc.returncode,
                           proc.stderr)


def download_massivekb_peaks(massivekb_filename: str) -> None:
    """
    Download all spectral data files listed in the given MassIVE-KB metadata
    file.

    Peak files will be stored in the `data/peak/{dataset}/{filename}`
    directories.
    Existing peak files will _not_ be downloaded again.

    Parameters
    ----------
    massivekb_filename : str
        The metadata file name.
    """
    filenames = pd.read_csv(massivekb_filename, sep='\t', usecols=['filename'],
                            squeeze=True).unique()
    logger.info('Download peak files from MassIVE')
    joblib.Parallel(n_jobs=-1)(joblib.delayed(download_massive_file)(filename)
                               for filename in filenames)


def generate_pairs_positive(metadata_filename: str) -> None:
    """
    Generate index pairs for positive training pairs for the given metadata
    file.

    The positive training pairs consist of all pairs with the same peptide
    sequence in the metadata. Identity pairs are included.
    Pairs of row numbers in the metadata file for each positive pair are stored
    in Parquet file `{metadata_filename}_pairs_pos.parquet`.
    If this file already exists it will _not_ be recreated.

    Parameters
    ----------
    metadata_filename : str
        The metadata file name. Should be a Parquet file.
    """
    pairs_filename = metadata_filename.replace('.parquet', '_pairs_pos.npy')
    if not os.path.isfile(pairs_filename):
        logger.info('Generate positive pair indexes for metadata file %s',
                    metadata_filename)
        metadata = pd.read_parquet(metadata_filename,
                                   columns=['sequence', 'charge'])
        metadata['row_num'] = range(len(metadata.index))
        same_row_nums = metadata.groupby(
            ['sequence', 'charge'], as_index=False, sort=False)['row_num']
        logger.debug('Save positive pair indexes to %s', pairs_filename)
        np.save(pairs_filename, np.asarray(
            [[p1, p2] for p1, p2 in itertools.chain(*(same_row_nums.apply(
                functools.partial(itertools.combinations, r=2))))],
            dtype=np.uint32))


def generate_pairs_negative(metadata_filename: str,
                            mz_tolerance: float) -> None:
    """
    Generate index pairs for negative training pairs for the given metadata
    file.

    The negative training pairs consist of all pairs with a different peptide
    sequence and a precursor m/z difference smaller than the given m/z
    tolerance in the metadata.
    Pairs of row numbers in the metadata file for each negative pair are stored
    in Parquet file `{metadata_filename}_pairs_neg.parquet`.
    If this file already exists it will _not_ be recreated.

    Parameters
    ----------
    metadata_filename : str
        The metadata file name. Should be a Parquet file.
    mz_tolerance : float
        Maximum precursor m/z tolerance in ppm for two PSMs to be considered a
        negative pair.
    """
    pairs_filename = metadata_filename.replace('.parquet', '_pairs_neg.npy')
    if not os.path.isfile(pairs_filename):
        logger.info('Generate negative pair indexes for metadata file %s',
                    metadata_filename)
        metadata = pd.read_parquet(metadata_filename,
                                   columns=['sequence', 'charge', 'mz'])
        metadata['row_num'] = range(len(metadata.index))
        metadata = (metadata.sort_values(['charge', 'mz'])
                    .reset_index(drop=True))
        row_nums = metadata['row_num'].values
        # List because Numba can't handle object (string) arrays.
        sequences = metadata['sequence'].tolist()
        mzs = metadata['mz'].values
        logger.debug('Save negative pair indexes to %s', pairs_filename)
        with warnings.catch_warnings():
            # FIXME: Deprecated reflected list in Numba should be resolved from
            #        version 0.46.0 onwards.
            #  https://numba.pydata.org/numba-doc/latest/reference/deprecation.html#deprecation-of-reflection-for-list-and-set-types
            warnings.simplefilter('ignore', nb.NumbaPendingDeprecationWarning)
            np.save(pairs_filename, np.asarray(
                [[p1, p2] for p1, p2 in _generate_pairs_negative(
                    row_nums, sequences, mzs, mz_tolerance)],
                dtype=np.uint32))


@nb.njit
def _generate_pairs_negative(row_nums: np.ndarray, sequences: List[str],
                             mzs: np.ndarray, mz_tolerance: float)\
        -> Iterator[Tuple[int, int]]:
    """
    Numba utility function to efficiently generate row numbers for negative
    pairs.

    Parameters
    ----------
    row_nums : np.ndarray
        A NumPy array of row numbers for each PSM.
    sequences : List[str]
        A list of peptide sequences for each PSM.
    mzs : np.ndarray
        A NumPy array of precursor m/z values for each PSM.
    mz_tolerance : float
        Maximum precursor m/z tolerance in ppm for two PSMs to be considered a
        negative pair.

    Returns
    -------
    Iterator[Tuple[int, int]]
        A generator of tuple with matching row numbers of the negative pairs.
    """
    for i in range(len(row_nums)):
        j = i + 1
        while (j < len(mzs) and
               abs(suu.mass_diff(mzs[i], mzs[j], False)) <= mz_tolerance):
            if sequences[i] != sequences[j]:
                yield row_nums[i], row_nums[j]
            j += 1
