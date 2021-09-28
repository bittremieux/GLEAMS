import functools
import itertools
import logging
import os
import re
import subprocess
from typing import Iterator, List, Optional, Tuple

import joblib
import numba as nb
import numpy as np
import pandas as pd
from spectrum_utils import spectrum as sus
from spectrum_utils import utils as suu


logger = logging.getLogger('gleams')


regex_non_alpha = re.compile(r'[^A-Za-z]+')
regex_mod = re.compile(r'\+\d+.\d+')


def convert_massivekb_metadata(massivekb_filename: str,
                               metadata_filename: str,
                               charges: Optional[Tuple[int]] = None) -> None:
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
    charges : Optional[Tuple[int]]
        Optional tuple of minimum and maximum precursor charge (both inclusive)
        to include, spectra with other precursor charges will be omitted.
    """
    if not os.path.isfile(metadata_filename):
        logger.info('Convert the MassIVE-KB metadata file')
        metadata = pd.read_csv(massivekb_filename, sep='\t', usecols=[
            'annotation', 'charge', 'filename', 'mz', 'scan'])
        if charges is not None:
            metadata = metadata[(metadata['charge'] >= charges[0]) &
                                (metadata['charge'] <= charges[1])].copy()
        metadata = metadata.rename(columns={'annotation': 'sequence'})
        dataset_filename = metadata['filename'].str.split('/').str
        metadata['dataset'] = dataset_filename[0]
        metadata['filename'] = dataset_filename[-1]
        metadata['scan'] = metadata['scan'].astype(np.int64)
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

    If metadata files corresponding to the training/validation/test splits
    already exist the splits will _not_ be recreated.

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
        metadata.loc[selected_val].reset_index().to_parquet(filename_val)
        datasets = datasets.drop(selected_val)
    if num_test > 0:
        selected_test = _select_datasets(datasets, num_test, abs_tol)
        logger.debug('Save test metadata file to %s', filename_test)
        metadata.loc[selected_test].reset_index().to_parquet(filename_test)
        datasets = datasets.drop(selected_test)
    logger.debug('Save train metadata file to %s', filename_train)
    metadata.loc[datasets.index].reset_index().to_parquet(filename_train)


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
    # noinspection PyTypeChecker
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


def generate_pairs_positive(metadata_filename: str,
                            charges: Tuple[int]) -> None:
    """
    Generate index pairs for positive training pairs for the given metadata
    file.

    The positive training pairs consist of all pairs with the same peptide
    sequence in the metadata, split by precursor charge. Identity pairs are
    included.
    Pairs of row numbers in the metadata file for each positive pair are stored
    in Parquet file `{metadata_filename}_pairs_pos_{charge}.parquet`.
    If these files already exists they will _not_ be recreated.

    Parameters
    ----------
    metadata_filename : str
        The metadata file name. Should be a Parquet file.
    charges : Tuple[int]
        Tuple of minimum and maximum precursor charge (both inclusive) to
        include, spectra with other precursor charges will be omitted.
    """
    metadata = (pd.read_parquet(metadata_filename,
                                columns=['sequence', 'charge'])
                .reset_index().dropna())
    metadata['sequence'] = metadata['sequence'].str.replace('I', 'L')
    for charge in np.arange(charges[0], charges[1] + 1):
        pairs_filename = metadata_filename.replace('.parquet',
                                                   f'_pairs_pos_{charge}.npy')
        if not os.path.isfile(pairs_filename):
            logger.info('Generate positive pair indexes for charge %d from '
                        'metadata file %s', charge, metadata_filename)
            same_row_nums = metadata[metadata['charge'] == charge].groupby(
                'sequence', sort=False)['index']
            logger.debug('Save positive pair indexes for charge %d to file %s',
                         charge, pairs_filename)
            np.save(pairs_filename, np.asarray(
                [[np.uint32(p1), np.uint32(p2)]
                 for p1, p2 in itertools.chain(*(same_row_nums.apply(
                    functools.partial(itertools.combinations, r=2))))],
                dtype=np.uint32))


def generate_pairs_negative(metadata_filename: str, charges: Tuple[int],
                            mz_tolerance: float, fragment_tolerance: float,
                            matching_fragments_threshold: float) \
        -> None:
    """
    Generate index pairs for negative training pairs for the given metadata
    file.

    The negative training pairs consist of all pairs with a different peptide
    sequence, a precursor m/z difference smaller than the given m/z tolerance,
    and mostly non-overlapping b and y ions, split by precursor charge.
    Pairs of row numbers in the metadata file for each negative pair are stored
    in Parquet file `{metadata_filename}_pairs_neg_{charge}.parquet`.
    If these files already exists they will _not_ be recreated.

    Parameters
    ----------
    metadata_filename : str
        The metadata file name. Should be a Parquet file.
    charges : Tuple[int]
        Tuple of minimum and maximum precursor charge (both inclusive) to
        include, spectra with other precursor charges will be omitted.
    mz_tolerance : float
        Maximum precursor m/z tolerance in ppm for two PSMs to be considered a
        negative pair.
    fragment_tolerance : float
        Maximum fragment m/z tolerance in Da for two fragments to be considered
        overlapping (to avoid overly similar negative pairs).
    matching_fragments_threshold : float
        Maximum ratio of matching fragments relative to the number of b and y
        ions of shortest peptide to be considered a negative pair (to avoid
        overly similar negative pairs).
    """
    metadata = (pd.read_parquet(metadata_filename,
                                columns=['sequence', 'charge', 'mz'])
                .reset_index().dropna()
                .sort_values(['charge', 'mz']).reset_index(drop=True))
    for charge in np.arange(charges[0], charges[1] + 1):
        pairs_filename = metadata_filename.replace('.parquet',
                                                   f'_pairs_neg_{charge}.npy')
        if not os.path.isfile(pairs_filename):
            logger.info('Generate negative pair indexes for charge %d from '
                        'metadata file %s', charge, metadata_filename)
            metadata_charge = metadata[metadata['charge'] == charge]
            # List because Numba can't handle object (string) arrays.
            sequences = nb.typed.List(metadata_charge['sequence']
                                      .str.replace('I', 'L')
                                      .apply(_remove_mod))
            fragments = nb.typed.List(metadata_charge['sequence']
                                      .apply(_get_theoretical_fragment_mzs))
            logger.debug('Save negative pair indexes for charge %d to file %s',
                         charge, pairs_filename)
            np.save(pairs_filename, np.fromiter(_generate_pairs_negative(
                metadata_charge['index'].values, metadata_charge['mz'].values,
                sequences, fragments, mz_tolerance, fragment_tolerance,
                matching_fragments_threshold), np.uint32)
                    .reshape((-1, 2)))


@functools.lru_cache(None)
def _get_theoretical_fragment_mzs(sequence: str) -> np.ndarray:
    """
    Get the theoretical b and y ion m/z values for the given peptide sequence.

    Parameters
    ----------
    sequence : str
        The peptide sequence for which b and y ion m/z values will be
        calculated.

    Returns
    -------
    np.ndarray
        An array of sorted m/z values of the b and y ions for the given peptide
        sequence.
    """
    # Correct for 0-based offsets required by spectrum_utils.
    mods, mod_pos_offset = {}, 1
    for match in re.finditer(regex_mod, sequence):
        mods[match.start() - mod_pos_offset] = float(match.group(0))
        mod_pos_offset += match.end() - match.start()
    # noinspection PyProtectedMember
    return np.asarray([fragment.calc_mz for fragment in
                       sus._get_theoretical_peptide_fragments(
                           _remove_mod(sequence), mods)])


@functools.lru_cache(None)
def _remove_mod(peptide: str) -> str:
    """
    Remove modifications indicated by a delta mass from the peptide sequence.

    Parameters
    ----------
    peptide : str
        The given peptide sequence string.

    Returns
    -------
    str
        The normalized peptide sequence, or None if the input was None.
    """
    return regex_non_alpha.sub('', peptide) if peptide is not None else None


@nb.njit
def _generate_pairs_negative(row_nums: np.ndarray, mzs: np.ndarray,
                             sequences: nb.typed.List,
                             fragments: nb.typed.List,
                             precursor_mz_tol: float, fragment_mz_tol: float,
                             matching_fragments_threshold: float) \
        -> Iterator[int]:
    """
    Numba utility function to efficiently generate row numbers for negative
    pairs.

    Parameters
    ----------
    row_nums : np.ndarray
        A NumPy array of row numbers for each PSM.
    mzs : np.ndarray
        A NumPy array of precursor m/z values for each PSM.
    sequences : nb.typed.List
        A list of peptide sequences for each PSM.
    fragments: nb.typed.List
        Theoretical fragments of the peptides corresponding to each PSM.
    precursor_mz_tol : float
        Maximum precursor m/z tolerance in ppm for two PSMs to be considered a
        negative pair.
    fragment_mz_tol : float
        Maximum fragment m/z tolerance in Da for two fragments to be considered
        overlapping.
    matching_fragments_threshold : float
        Maximum ratio of matching fragments relative to the number of b and y
        ions of shortest peptide to be considered a negative pair.

    Returns
    -------
    Iterator[int]
        A generator of row numbers for the negative pairs, with row numbers `i`
        and `i + 1` forming pairs.
    """
    for row_num1 in range(len(row_nums)):
        row_num2 = row_num1 + 1
        while (row_num2 < len(mzs) and
               (abs(suu.mass_diff(mzs[row_num1], mzs[row_num2], False))
                <= precursor_mz_tol)):
            if sequences[row_num1] != sequences[row_num2]:
                fragments1 = fragments[row_num1]
                fragments2 = fragments[row_num2]
                num_matching_fragments = 0
                for fragment1_i, fragment2 in zip(
                        np.searchsorted(fragments1, fragments2), fragments2):
                    fragment1_left = fragments1[max(0, fragment1_i - 1)]
                    fragment1_right = fragments1[min(fragment1_i,
                                                     len(fragments1) - 1)]
                    if ((abs(fragment1_left - fragment2) < fragment_mz_tol)
                            or (abs(fragment1_right - fragment2)
                                < fragment_mz_tol)):
                        num_matching_fragments += 1

                if num_matching_fragments < matching_fragments_threshold * min(
                        len(fragments1), len(fragments2)):
                    yield row_nums[row_num1]
                    yield row_nums[row_num2]

            row_num2 += 1
