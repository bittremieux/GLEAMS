import logging
import math
import os
from typing import Iterator, List, Optional, Tuple

import faiss
import fastcluster
import joblib
import numba as nb
import numpy as np
import pandas as pd
import scipy.sparse as ss
import tqdm
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
# noinspection PyProtectedMember
from sklearn.cluster._dbscan_inner import dbscan_inner
from sklearn.metrics import pairwise_distances

from gleams import config


logger = logging.getLogger('gleams')


def _check_ann_config() -> None:
    """
    Make sure that the configuration values adhere to the limitations imposed
    by running Faiss on a GPU.
    GPU indexes can only handle maximum 1024 probes and neighbors.
    https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU#limitations
    """
    if config.num_probe > 1024:
        logger.warning('Using num_probe=1024 (maximum supported value for '
                       'GPU-enabled ANN indexing), %d was supplied',
                       config.num_probe)
        config.num_probe = 1024
    if config.num_neighbors > 1024:
        logger.warning('Using num_neighbours=1024 (maximum supported value '
                       'for GPU-enabled ANN indexing), %d was supplied',
                       config.num_neighbors)
        config.num_neighbors = 1024


_check_ann_config()


def compute_pairwise_distances(embeddings_filename: str,
                               metadata_filename: str,
                               charges: Optional[Tuple[int]] = None) -> None:
    """
    Compute a pairwise distance matrix for the embeddings in the given file.

    Parameters
    ----------
    embeddings_filename : str
        NumPy file containing the embedding vectors for which to compute
        pairwise distances.
    metadata_filename : str
        Metadata file with precursor m/z information for all embeddings.
    charges : Optional[Tuple[int]]
        Optional tuple of minimum and maximum precursor charge (both inclusive)
        to include, spectra with other precursor charges will be omitted.
    """
    cluster_dir = os.path.join(os.environ['GLEAMS_HOME'], 'data', 'cluster')
    if not os.path.exists(cluster_dir):
        os.mkdir(cluster_dir)
    ann_dir = os.path.join(cluster_dir, 'ann')
    if not os.path.exists(ann_dir):
        os.mkdir(ann_dir)
    index_filename = os.path.splitext(
        os.path.basename(embeddings_filename))[0].replace('embed_', 'ann_')
    index_filename = os.path.join(ann_dir, index_filename + '_{}_{}.faiss')
    dist_filename = (os.path.splitext(
        os.path.basename(embeddings_filename))[0].replace('embed_', 'dist_'))
    dist_filename = os.path.join(cluster_dir, f'{dist_filename}.npz')
    neighbors_filename = (dist_filename.replace('dist_', 'neighbors_{}_')
                                       .replace('.npz', '.npy'))
    if os.path.isfile(dist_filename):
        return
    metadata = pd.read_parquet(metadata_filename).sort_values(['charge', 'mz'])
    metadata = metadata[metadata['charge'].isin(
        np.arange(charges[0], charges[1] + 1))].reset_index()
    embeddings = np.load(embeddings_filename)
    embeddings = embeddings[metadata['index']]
    min_mz, max_mz = metadata['mz'].min(), metadata['mz'].max()
    mz_splits = np.arange(
        math.floor(min_mz / config.mz_interval) * config.mz_interval,
        math.ceil(max_mz / config.mz_interval) * config.mz_interval,
        config.mz_interval)
    # Create the ANN indexes (if this hasn't been done yet).
    _build_ann_index(index_filename, embeddings, metadata[['charge', 'mz']],
                     mz_splits)
    # Calculate pairwise distances.
    logging.info('Compute pairwise distances between neighboring embeddings '
                 '(%d embeddings, %d neighbors)', len(metadata),
                 config.num_neighbors)
    if len(metadata) > np.iinfo(np.int64).max:
        raise OverflowError('Too many embedding indexes to fit into int64')
    if (not os.path.isfile(neighbors_filename.format('data')) or
            not os.path.isfile(neighbors_filename.format('indices')) or
            not os.path.isfile(neighbors_filename.format('indptr'))):
        max_num_embeddings = len(metadata) * config.num_neighbors
        dtype = (np.int32 if max_num_embeddings < np.iinfo(np.int32).max
                 else np.int64)
        distances = np.zeros(max_num_embeddings, np.float32)
        indices = np.zeros(max_num_embeddings, dtype)
        indptr = np.zeros(len(metadata) + 1, dtype)
        with tqdm.tqdm(total=metadata['charge'].nunique() * len(mz_splits),
                       desc='Distances calculated', unit='index') as pbar:
            for charge, precursors_charge in (metadata[['charge', 'mz']]
                                              .groupby('charge')):
                for mz in mz_splits:
                    _dist_mz_interval(
                        index_filename, embeddings, precursors_charge['mz'],
                        distances, indices, indptr, charge, mz)
                    pbar.update(1)
        distances, indices = distances[:indptr[-1]], indices[:indptr[-1]]
        np.save(neighbors_filename.format('data'), distances)
        np.save(neighbors_filename.format('indices'), indices)
        np.save(neighbors_filename.format('indptr'), indptr)
    else:
        distances = np.load(neighbors_filename.format('data'))
        indices = np.load(neighbors_filename.format('indices'))
        indptr = np.load(neighbors_filename.format('indptr'))
    embeddings_filename = os.path.join(
        cluster_dir, os.path.basename(embeddings_filename))
    metadata_filename = os.path.join(
        cluster_dir, os.path.basename(metadata_filename))
    # Convert to a sparse pairwise distance matrix. This matrix might not be
    # entirely symmetrical, but that shouldn't matter too much.
    logger.debug('Construct pairwise distance matrix')
    pairwise_dist_matrix = ss.csr_matrix(
        (distances, indices, indptr), (len(metadata), len(metadata)),
        np.float32, False)
    logger.debug('Save the pairwise distance matrix to file %s', dist_filename)
    ss.save_npz(dist_filename, pairwise_dist_matrix, False)
    # Sort the embeddings and metadata in the same order as the pairwise
    # distance matrix.
    logger.debug('Save the reordered embeddings to file %s',
                 embeddings_filename)
    np.save(embeddings_filename, embeddings)
    logger.debug('Save the metadata to file %s', metadata_filename)
    metadata.drop(columns='index', inplace=True)
    metadata.to_parquet(metadata_filename, index=False)
    logger.debug('Clean up temporary pairwise distance files %s',
                 neighbors_filename)
    os.remove(neighbors_filename.format('data'))
    os.remove(neighbors_filename.format('indices'))
    os.remove(neighbors_filename.format('indptr'))


def _build_ann_index(index_filename: str, embeddings: np.ndarray,
                     precursors: pd.DataFrame, mz_splits: np.ndarray) -> None:
    """
    Create ANN indexes for the given embedding vectors.

    Vectors will be split over multiple ANN indexes based on the given m/z
    interval.

    Parameters
    ----------
    index_filename: str
        Base file name of the ANN index. Separate indexes for the given m/z
        splits will be created.
    embeddings: np.ndarray
        The embedding vectors to build the ANN index.
    precursors : pd.DataFrame
        Precursor charges and m/z's corresponding to the embedding vectors used
        to split the embeddings over multiple ANN indexes per charge and m/z
        interval.
    mz_splits: np.ndarray
        M/z splits used to create separate ANN indexes.
    """
    logger.debug('Use %d GPUs for ANN index construction',
                 faiss.get_num_gpus())
    # Create separate indexes per precursor charge and with precursor m/z in
    # the specified intervals.
    with tqdm.tqdm(total=precursors['charge'].nunique() * len(mz_splits),
                   desc='Indexes built', unit='index') as progressbar:
        for charge, precursors_charge in precursors.groupby('charge'):
            for mz in mz_splits:
                progressbar.update(1)
                if os.path.isfile(index_filename.format(charge, mz)):
                    continue
                # Create an ANN index using Euclidean distance
                # for fast NN queries.
                start_i, stop_i = _get_precursor_mz_interval_ids(
                    precursors_charge['mz'].values, mz, config.mz_interval,
                    config.precursor_tol_mode, config.precursor_tol_mass)
                index_embeddings_ids = (precursors_charge.index
                                        .values[start_i:stop_i])
                num_index_embeddings = len(index_embeddings_ids)
                # Figure out a decent value for the num_list hyperparameter
                # based on the number of embeddings.
                # Rules of thumb from the Faiss wiki:
                # https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index#how-big-is-the-dataset
                if num_index_embeddings == 0:
                    continue
                if num_index_embeddings < 10e2:
                    # Use a brute-force index instead of an ANN index
                    # when there are only a few items.
                    num_list = -1
                elif num_index_embeddings < 10e5:
                    num_list = 2**math.floor(math.log2(
                        num_index_embeddings / 39))
                elif num_index_embeddings < 10e6:
                    num_list = 2**16
                elif num_index_embeddings < 10e7:
                    num_list = 2**18
                else:
                    num_list = 2**20
                    if num_index_embeddings > 10e8:
                        logger.warning('More than 1B embeddings to be indexed,'
                                       ' consider decreasing the ANN size')
                logger.debug('Build the ANN index for precursor charge %d and '
                             'precursor m/z %d–%d (%d embeddings, %d lists)',
                             charge, int(mz), int(mz + config.mz_interval),
                             num_index_embeddings, num_list)
                # Create a suitable index and compute cluster centroids.
                if num_list <= 0:
                    index = faiss.IndexIDMap(
                        faiss.IndexFlatL2(config.embedding_size))
                else:
                    index = faiss.IndexIVFFlat(
                        faiss.IndexFlatL2(config.embedding_size),
                        config.embedding_size, num_list, faiss.METRIC_L2)
                index_embeddings = embeddings[index_embeddings_ids]
                # noinspection PyArgumentList
                index.train(index_embeddings)
                # Add the embeddings to the index in batches.
                logger.debug('Add %d embeddings to the ANN index',
                             num_index_embeddings)
                batch_size = min(num_index_embeddings, config.batch_size_add)
                for batch_start in range(0, num_index_embeddings, batch_size):
                    batch_stop = min(batch_start + batch_size,
                                     num_index_embeddings)
                    # noinspection PyArgumentList
                    index.add_with_ids(
                        index_embeddings[batch_start:batch_stop],
                        index_embeddings_ids[batch_start:batch_stop])
                # Save the index to disk.
                logger.debug('Save the ANN index to file %s',
                             index_filename.format(charge, mz))
                faiss.write_index(index, index_filename.format(charge, mz))
                index.reset()


def _dist_mz_interval(index_filename: str, embeddings: np.ndarray,
                      precursor_mzs: pd.Series, distances: np.ndarray,
                      indices: np.ndarray, indptr: np.ndarray, charge: int,
                      mz: int) -> None:
    """
    Compute distances to the nearest neighbors for the given precursor m/z
    interval.

    Parameters
    ----------
    index_filename: str
        Base file name of the ANN index. The specific index for the given m/z
        will be used.
    embeddings: np.ndarray
        The embedding vectors.
    precursor_mzs: pd.Series
        Precursor m/z's corresponding to the embedding vectors.
    distances : np.ndarray
        The nearest neighbor distances.
        See `scipy.sparse.csr_matrix` (`data`).
    indices : np.ndarray
        The column indices for the nearest neighbor distances.
        See `scipy.sparse.csr_matrix`.
    indptr : np.ndarray
        The index pointers for the nearest neighbor distances.
        See `scipy.sparse.csr_matrix`.
    charge : int
        The active precursor charge to load the ANN index.
    mz : int
        The active precursor m/z split to load the ANN index.
    """
    if not os.path.isfile(index_filename.format(charge, mz)):
        return
    index = _load_ann_index(index_filename.format(charge, mz))
    start_i, stop_i = _get_precursor_mz_interval_ids(
        precursor_mzs.values, mz, config.mz_interval, None, 0)
    for batch_start in range(start_i, stop_i, config.batch_size_dist):
        batch_stop = min(batch_start + config.batch_size_dist, stop_i)
        batch_ids = precursor_mzs.index.values[batch_start:batch_stop]
        # Find nearest neighbors using ANN index searching.
        # noinspection PyArgumentList
        nn_dists, nn_idx_ann = index.search(
            embeddings[batch_ids], config.num_neighbors_ann)
        # Filter the neighbors based on the precursor m/z tolerance and assign
        # distances.
        _filter_neighbors_mz(
            precursor_mzs.values, precursor_mzs.index.values, batch_start,
            batch_stop, config.precursor_tol_mass, config.precursor_tol_mode,
            nn_dists, nn_idx_ann, config.num_neighbors, distances, indices,
            indptr)
    index.reset()


def _load_ann_index(index_filename: str) -> faiss.Index:
    """
    Load the ANN index from the given file.

    Parameters
    ----------
    index_filename : str
        The ANN index filename.

    Returns
    -------
    faiss.Index
        The Faiss `Index`.
    """
    index = faiss.read_index(index_filename)
    # IndexIVF has a `nprobe` hyperparameter, flat indexes don't.
    if hasattr(index, 'nprobe'):
        index.nprobe = min(math.ceil(index.nlist / 2), config.num_probe)
    return index


@nb.njit
def _get_precursor_mz_interval_ids(precursor_mzs: np.ndarray, start_mz: float,
                                   mz_window: float,
                                   precursor_tol_mode: Optional[str],
                                   precursor_tol_mass: float) -> \
        Tuple[int, int]:
    """
    Get the IDs of the embeddings falling within the specified precursor m/z
    interval (taking a small margin for overlapping intervals into account).

    Parameters
    ----------
    precursor_mzs : np.ndarray
        Array of sorted precursor m/z's.
    start_mz : float
        The lower end of the m/z interval.
    mz_window : float
        The width of the m/z interval.
    precursor_tol_mode : Optional[str]
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
    precursor_tol_mass : float
        The value of the precursor m/z tolerance.

    Returns
    -------
    Tuple[int, int]
        The start and stop index of the embedding identifiers falling within
        the specified precursor m/z interval.
    """
    if precursor_tol_mode == 'Da':
        margin = precursor_tol_mass
    elif precursor_tol_mode == 'ppm':
        margin = precursor_tol_mass * start_mz / 10**6
    else:
        margin = 0
    if margin > 0:
        margin = max(margin, mz_window / 100)
    idx = np.searchsorted(precursor_mzs, [start_mz - margin,
                                          start_mz + mz_window + margin])
    return idx[0], idx[1]


@nb.njit
def _filter_neighbors_mz(
        precursor_mzs: np.ndarray, idx: np.ndarray, batch_start: int,
        batch_stop: int, precursor_tol_mass: float, precursor_tol_mode: str,
        nn_dists: np.ndarray, nn_idx_ann: np.ndarray,
        num_neighbors: int, distances: np.ndarray, indices: np.ndarray,
        indptr: np.ndarray) -> None:
    """
    Filter ANN neighbor indexes by precursor m/z tolerances and assign
    pairwise distances.

    Parameters
    ----------
    precursor_mzs : np.ndarray
        Precursor m/z's corresponding to the embeddings.
    idx : np.ndarray
        The indexes corresponding to the embeddings.
    batch_start, batch_stop : int
        The indexes in the precursor m/z's of the current batch.
    precursor_tol_mass : float
        The precursor tolerance mass for embeddings to be considered as
        neighbors.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
    nn_dists : np.ndarray
        Distances of the nearest neighbors.
    nn_idx_ann : np.ndarray
        Indexes of the nearest neighbors.
    num_neighbors : int
        The (maximum) number of neighbors to set for each embedding.
    distances : np.ndarray
        The nearest neighbor distances. See `scipy.sparse.csr_matrix` (`data`).
    indices : np.ndarray
        The column indices for the nearest neighbor distances. See
        `scipy.sparse.csr_matrix`.
    indptr : np.ndarray
        The index pointers for the nearest neighbor distances. See
        `scipy.sparse.csr_matrix`.
    """
    nn_idx_mz = _get_neighbors_idx(
        precursor_mzs, idx, batch_start, batch_stop, precursor_tol_mass,
        precursor_tol_mode)
    for i, idx_ann, idx_mz, dists in zip(
            idx[batch_start:batch_stop], nn_idx_ann, nn_idx_mz, nn_dists):
        mask = _intersect_idx_ann_mz(idx_ann, idx_mz, num_neighbors)
        indptr[i + 1] = indptr[i] + len(mask)
        distances[indptr[i]:indptr[i + 1]] = dists[mask]
        indices[indptr[i]:indptr[i + 1]] = idx_ann[mask]


@nb.njit
def _get_neighbors_idx(mzs: np.ndarray, idx: np.ndarray, start_i: int,
                       stop_i: int, precursor_tol_mass: float,
                       precursor_tol_mode: str) -> List[np.ndarray]:
    """
    Filter nearest neighbor candidates on precursor m/z.

    Parameters
    ----------
    mzs : np.ndarray
        The precursor m/z's of the nearest neighbor candidates.
    idx : np.ndarray
        The indexes of the nearest neighbor candidates.
    start_i, stop_i : int
        Indexes used to slice the values to be considered in the batch
        (inclusive start_i, exclusive stop_i).
    precursor_tol_mass : float
        The tolerance for vectors to be considered as neighbors.
    precursor_tol_mode : str
        The unit of the tolerance ('Da' or 'ppm').

    Returns
    -------
    List[np.ndarray]
        A list of sorted NumPy arrays with the indexes of the nearest neighbor
        candidates for each item.
    """
    batch_mzs = mzs[start_i:stop_i]
    if precursor_tol_mode == 'Da':
        min_mz = batch_mzs[0] - precursor_tol_mass
        max_mz = batch_mzs[-1] + precursor_tol_mass
    elif precursor_tol_mode == 'ppm':
        min_mz = batch_mzs[0] - batch_mzs[0] * precursor_tol_mass / 10**6
        max_mz = batch_mzs[-1] + batch_mzs[-1] * precursor_tol_mass / 10**6
    else:
        raise ValueError('Unknown precursor tolerance filter')
    batch_mzs = batch_mzs.reshape((-1, 1))
    match_i = np.searchsorted(mzs, [min_mz, max_mz])
    match_mzs = mzs[match_i[0]:match_i[1]].reshape((1, -1))
    if precursor_tol_mode == 'Da':
        masks = np.abs(batch_mzs - match_mzs) < precursor_tol_mass
    elif precursor_tol_mode == 'ppm':
        masks = (np.abs(batch_mzs - match_mzs) / match_mzs * 10**6
                 < precursor_tol_mass)
    match_idx = idx[match_i[0]:match_i[1]]
    # noinspection PyUnboundLocalVariable
    return [np.sort(match_idx[mask]) for mask in masks]


@nb.njit
def _intersect_idx_ann_mz(idx_ann: np.ndarray, idx_mz: np.ndarray,
                          max_neighbors: int) -> np.ndarray:
    """
    Find the intersection between identifiers from ANN filtering and precursor
    m/z filtering.

    Parameters
    ----------
    idx_ann : np.ndarray
        Identifiers from ANN filtering.
    idx_mz : np.ndarray
        SORTED identifiers from precursor m/z filtering.
    max_neighbors : int
        The maximum number of best matching neighbors to retain.

    Returns
    -------
    np.ndarray
        A mask to select the joint identifiers in the `idx_ann` array.
    """
    i_mz, idx_ann_order, idx = 0, np.argsort(idx_ann), []
    for i_order, i_ann in enumerate(idx_ann_order):
        if idx_ann[i_ann] != -1:
            while i_mz < len(idx_mz) and idx_mz[i_mz] < idx_ann[i_ann]:
                i_mz += 1
            if i_mz == len(idx_mz):
                break
            if idx_ann[i_ann] == idx_mz[i_mz]:
                idx.append(idx_ann_order[i_order])
                i_mz += 1
    idx = np.asarray(idx)
    return (idx if max_neighbors >= len(idx)
            else np.partition(idx, max_neighbors)[:max_neighbors])


def cluster(distances_filename: str, metadata_filename: str):
    """
    DBSCAN clustering of the embeddings based on a pairwise distance matrix.

    Parameters
    ----------
    distances_filename : str
        Precomputed pairwise distance matrix file to use for the DBSCAN
        clustering.
    metadata_filename : str
        Metadata file with precursor m/z information for all embeddings.
    """
    clusters_filename = (distances_filename.replace('dist_', 'clusters_')
                                           .replace('.npz', '.npy'))
    if os.path.isfile(clusters_filename):
        return

    # DBSCAN clustering of the embeddings.
    logger.info('DBSCAN clustering (eps=%.4f, min_samples=%d) of precomputed '
                'pairwise distance matrix %s', config.eps, config.min_samples,
                distances_filename)
    # Reimplement DBSCAN preprocessing to avoid unnecessary memory consumption.
    pairwise_dist_matrix = ss.load_npz(distances_filename)
    # Find the eps-neighborhoods for all points.
    logger.debug('Find the eps-neighborhoods for all points (eps=%.4f)',
                 config.eps)
    mask = pairwise_dist_matrix.data <= config.eps
    indices = pairwise_dist_matrix.indices[mask].astype(np.intp)
    # noinspection PyTypeChecker
    indptr = np.zeros(len(mask) + 1, dtype=np.int64)
    np.cumsum(mask, out=indptr[1:])
    indptr = indptr[pairwise_dist_matrix.indptr]
    neighborhoods = np.split(indices, indptr[1:-1])
    # Initially, all samples are noise.
    # (Memmap for shared memory multiprocessing.)
    cluster_labels = np.lib.format.open_memmap(
        clusters_filename, mode='w+', dtype=np.intp,
        shape=pairwise_dist_matrix.shape[0])
    cluster_labels.fill(-1)
    # A list of all core samples found.
    n_neighbors = np.fromiter(map(len, neighborhoods), np.uint32)
    core_samples = n_neighbors >= config.min_samples
    # Run Scikit-Learn DBSCAN.
    logger.debug('Run Scikit-Learn DBSCAN inner.')
    neighborhoods_arr = np.empty(len(neighborhoods), dtype=np.object)
    neighborhoods_arr[:] = neighborhoods
    dbscan_inner(core_samples, neighborhoods_arr, cluster_labels)

    # Refine initial clusters to make sure spectra within a cluster don't have
    # an excessive precursor m/z difference.
    precursor_mzs = (pd.read_parquet(metadata_filename, columns=['mz'])
                     .squeeze().values)
    logger.debug('Sort cluster labels in ascending order.')
    order = np.argsort(cluster_labels)
    reverse_order = np.argsort(order)
    cluster_labels[:] = cluster_labels[order]
    precursor_mzs = precursor_mzs[order]
    logger.debug('Finetune %d initial cluster assignments to not exceed %d %s '
                 'precursor m/z tolerance', cluster_labels[-1] + 1,
                 config.precursor_tol_mass, config.precursor_tol_mode)
    if cluster_labels[-1] == -1:     # Only noise samples.
        cluster_labels.fill(-1)
    else:
        group_idx = nb.typed.List(_get_cluster_group_idx(cluster_labels))
        n_clusters = nb.typed.List(joblib.Parallel(n_jobs=-1)(
            joblib.delayed(_postprocess_cluster)
            (cluster_labels[start_i:stop_i], precursor_mzs[start_i:stop_i],
             config.precursor_tol_mass, config.precursor_tol_mode,
             config.min_samples) for start_i, stop_i in group_idx))
        _assign_unique_cluster_labels(cluster_labels, group_idx,
                                      n_clusters, config.min_samples)
        cluster_labels = cluster_labels[reverse_order]
    logger.debug('%d unique clusters after precursor m/z finetuning',
                 np.amax(cluster_labels) + 1)


@nb.njit
def _get_cluster_group_idx(clusters: np.ndarray) -> Iterator[Tuple[int, int]]:
    """
    Get start and stop indexes for unique cluster labels.

    Parameters
    ----------
    clusters : np.ndarray
        The ordered cluster labels (noise points are -1).

    Returns
    -------
    Iterator[Tuple[int, int]]
        Tuples with the start index (inclusive) and end index (exclusive) of
        the unique cluster labels.
    """
    start_i = 0
    while clusters[start_i] == -1 and start_i < clusters.shape[0]:
        start_i += 1
    stop_i = start_i
    while stop_i < clusters.shape[0]:
        start_i, label = stop_i, clusters[stop_i]
        while stop_i < clusters.shape[0] and clusters[stop_i] == label:
            stop_i += 1
        yield start_i, stop_i


def _postprocess_cluster(cluster_labels: np.ndarray, cluster_mzs: np.ndarray,
                         precursor_tol_mass: float, precursor_tol_mode: str,
                         min_samples: int) -> int:
    """
    Agglomerative clustering of the precursor m/z's within each initial
    cluster to avoid that spectra within a cluster have an excessive precursor
    m/z difference.

    Parameters
    ----------
    cluster_labels : np.ndarray
        Array in which to write the cluster labels.
    cluster_mzs : np.ndarray
        Precursor m/z's of the samples in a single initial cluster.
    precursor_tol_mass : float
        Maximum precursor mass tolerance for points to be clustered together.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
    min_samples : int
        The minimum number of samples in a cluster.

    Returns
    -------
    int
        The number of clusters after splitting on precursor m/z.
    """
    cluster_labels.fill(-1)
    # No splitting needed if there are too few items in cluster.
    # This seems to happen sometimes despite that DBSCAN requires a higher
    # `min_samples`.
    if cluster_labels.shape[0] < min_samples:
        n_clusters = 0
    else:
        cluster_mzs = cluster_mzs.reshape(-1, 1)
        # Pairwise differences in Dalton.
        pairwise_mz_diff = pairwise_distances(cluster_mzs)
        if precursor_tol_mode == 'ppm':
            pairwise_mz_diff = pairwise_mz_diff / cluster_mzs * 10**6
        # Group items within the cluster based on their precursor m/z.
        # Precursor m/z's within a single group can't exceed the specified
        # precursor m/z tolerance (`distance_threshold`).
        # Subtract 1 because fcluster starts with cluster label 1 instead of 0
        # (like scikit-learn does).
        cluster_assignments = fcluster(
            fastcluster.linkage(
                squareform(pairwise_mz_diff, checks=False), 'complete'),
            precursor_tol_mass, 'distance') - 1
        n_clusters = cluster_assignments.max() + 1
        # Update cluster assignments.
        if n_clusters == 1:
            # Single homogeneous cluster.
            cluster_labels.fill(0)
        elif n_clusters == cluster_mzs.shape[0]:
            # Only singletons.
            n_clusters = 0
        else:
            cluster_assignments = cluster_assignments.reshape(1, -1)
            label, labels = 0, np.arange(n_clusters).reshape(-1, 1)
            # noinspection PyTypeChecker
            for mask in cluster_assignments == labels:
                if mask.sum() >= min_samples:
                    cluster_labels[mask] = label
                    label += 1
            n_clusters = label
    return n_clusters


@nb.njit
def _assign_unique_cluster_labels(cluster_labels: np.ndarray,
                                  group_idx: nb.typed.List,
                                  n_clusters: nb.typed.List,
                                  min_samples: int) -> None:
    """
    Make sure all cluster labels are unique after potential splitting of
    clusters to avoid excessive precursor m/z differences.

    Parameters
    ----------
    cluster_labels : np.ndarray
        Cluster labels per cluster grouping.
    group_idx : nb.typed.List[Tuple[int, int]]
        Tuples with the start index (inclusive) and end index (exclusive) of
        the cluster groupings.
    n_clusters: nb.typed.List[int]
        The number of clusters per cluster grouping.
    min_samples : int
        The minimum number of samples in a cluster.
    """
    current_label = 0
    for (start_i, stop_i), n_cluster in zip(group_idx, n_clusters):
        if n_cluster > 0 and stop_i - start_i >= min_samples:
            cluster_labels[start_i:stop_i] += current_label
            current_label += n_cluster
        else:
            cluster_labels[start_i:stop_i].fill(-1)


def get_cluster_medoids(clusters_filename: str, distances_filename: str):
    """
    Get indexes of the cluster representative spectra (medoids).

    Parameters
    ----------
    clusters_filename : str
        Cluster label assignments file.
    distances_filename : str
        Precomputed pairwise distance matrix file to use for the DBSCAN
        clustering.

    Returns
    -------
    Optional[np.ndarray]
        The indexes of the medoid elements for all non-noise clusters, or None
        if only noise clusters are present.
    """
    pairwise_dist_matrix = ss.load_npz(distances_filename)
    return _get_cluster_medoids(
        np.load(clusters_filename), pairwise_dist_matrix.indptr,
        pairwise_dist_matrix.indices, pairwise_dist_matrix.data)


@nb.njit(parallel=True)
def _get_cluster_medoids(clusters: np.ndarray,
                         pairwise_indptr: np.ndarray,
                         pairwise_indices: np.ndarray,
                         pairwise_data: np.ndarray) \
        -> Optional[np.ndarray]:
    """
    Get indexes of the cluster representative spectra (medoids).

    Parameters
    ----------
    clusters : np.ndarray
        Cluster label assignments.
    pairwise_indptr : np.ndarray
        The index pointers for the nearest neighbor distances. See
        `scipy.sparse.csr_matrix`.
    pairwise_indices : np.ndarray
        The column indices for the nearest neighbor distances. See
        `scipy.sparse.csr_matrix`.
    pairwise_data : np.ndarray
        The nearest neighbor distances. See `scipy.sparse.csr_matrix` (`data`).

    Returns
    -------
    Optional[np.ndarray]
        The indexes of the medoid elements for all non-noise clusters, or None
        if only noise clusters are present.
    """
    order, min_i = np.argsort(clusters), 0
    while min_i < clusters.shape[0] and clusters[order[min_i]] == -1:
        min_i += 1
    # Only noise clusters.
    if min_i == clusters.shape[0]:
        return None
    # Find the indexes of the representatives for each unique cluster.
    cluster_idx, max_i = [], min_i
    while max_i < order.shape[0]:
        while (max_i < order.shape[0] and
               clusters[order[min_i]] == clusters[order[max_i]]):
            max_i += 1
        cluster_idx.append((min_i, max_i))
        min_i = max_i
    representatives = np.empty(len(cluster_idx), np.uint)
    for i in nb.prange(len(cluster_idx)):
        representatives[i] = _get_cluster_medoid_index(
            order[cluster_idx[i][0]:cluster_idx[i][1]], pairwise_indptr,
            pairwise_indices, pairwise_data)
    return representatives


@nb.njit(fastmath=True)
def _get_cluster_medoid_index(cluster_mask: np.ndarray,
                              pairwise_indptr: np.ndarray,
                              pairwise_indices: np.ndarray,
                              pairwise_data: np.ndarray) -> int:
    """
    Get the index of the cluster medoid element.

    Parameters
    ----------
    cluster_mask : np.ndarray
        Indexes of the items belonging to the current cluster.
    pairwise_indptr : np.ndarray
        The index pointers for the nearest neighbor distances. See
        `scipy.sparse.csr_matrix`.
    pairwise_indices : np.ndarray
        The column indices for the nearest neighbor distances. See
        `scipy.sparse.csr_matrix`.
    pairwise_data : np.ndarray
        The nearest neighbor distances. See `scipy.sparse.csr_matrix` (`data`).

    Returns
    -------
    int
        The index of the cluster's medoid element.
    """
    if len(cluster_mask) <= 2:
        # Pairwise distances will be identical.
        return cluster_mask[0]
    min_i, min_avg = 0, np.inf
    for row_i in range(cluster_mask.shape[0]):
        indices = pairwise_indices[pairwise_indptr[cluster_mask[row_i]]:
                                   pairwise_indptr[cluster_mask[row_i] + 1]]
        data = pairwise_data[pairwise_indptr[cluster_mask[row_i]]:
                             pairwise_indptr[cluster_mask[row_i] + 1]]
        col_i = np.asarray([i for cm in cluster_mask
                            for i, ind in enumerate(indices) if cm == ind])
        row_avg = np.mean(data[col_i])
        if row_avg < min_avg:
            min_i, min_avg = row_i, row_avg
    return cluster_mask[min_i]
