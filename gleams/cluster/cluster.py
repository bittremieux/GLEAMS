import gc
import logging
import math
import os
import warnings
from typing import Iterator, Optional, Tuple

import fastcluster
import joblib
import numba as nb
import numpy as np
import pandas as pd
import scipy.sparse as ss
import spectrum_utils.utils as suu
import tqdm
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist


logger = logging.getLogger('gleams')


def cluster(embeddings_filename: str, metadata_filename: str,
            clusters_filename: str, precursor_tol_mass: float,
            precursor_tol_mode: str, linkage: str, distance_threshold: float,
            charges: Optional[Tuple[int]] = None) -> None:
    """
    Cluster the GLEAMS embeddings.

    Parameters
    ----------
    embeddings_filename : str
        NumPy file containing the embedding vectors to cluster.
    metadata_filename : str
        Metadata file with precursor m/z information for all embeddings.
    clusters_filename : str
        File name to export the cluster labels.
    precursor_tol_mass : float
        The value of the precursor m/z tolerance.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
    linkage : str
        Linkage method to calculate the cluster distances. See
        `scipy.cluster.hierarchy.linkage` for possible options.
    distance_threshold : float
        The maximum linkage distance threshold during clustering.
    charges : Optional[Tuple[int]]
        Optional tuple of minimum and maximum precursor charge (both inclusive)
        to include, spectra with other precursor charges will be omitted.
    """
    if os.path.isfile(clusters_filename):
        warnings.warn('The clustering results file already exists and was '
                      'not recomputed')
        return
    clusters_dir = os.path.dirname(clusters_filename)
    if not os.path.exists(clusters_dir):
        os.mkdir(clusters_dir)
    # Sort the metadata by increasing precursor m/z for easy subsetting.
    metadata = (pd.read_parquet(metadata_filename, columns=['charge', 'mz'])
                .reset_index().sort_values(['charge', 'mz']))
    metadata = metadata[metadata['charge'].isin(
        np.arange(charges[0], charges[1] + 1))]
    embeddings = np.load(embeddings_filename, mmap_mode='r')
    # Cluster per contiguous block of precursor m/z's (relative to the
    # precursor m/z threshold).
    logging.info('Cluster %d embeddings using %s linkage and distance '
                 'threshold %.3f', len(metadata), linkage, distance_threshold)
    # Initially, all samples are noise. (Memmap for memory efficiency.)
    # noinspection PyUnresolvedReferences
    cluster_labels = np.lib.format.open_memmap(
        clusters_filename, mode='w+', dtype=np.int32,
        shape=(embeddings.shape[0],))
    cluster_labels.fill(-1)
    max_label = 0
    with tqdm.tqdm(total=len(metadata), desc='Clustering', unit='embedding',
                   smoothing=0) as pbar:
        for _, metadata_charge in metadata.groupby('charge'):
            idx = metadata_charge['index'].values
            mz = metadata_charge['mz'].values
            splits = _get_precursor_mz_splits(mz, precursor_tol_mass,
                                              precursor_tol_mode)
            # Per-split cluster labels.
            joblib.Parallel(n_jobs=-1, backend='threading')(
                joblib.delayed(_cluster_interval)
                (embeddings, idx, mz, cluster_labels, splits[i], splits[i+1],
                 linkage, distance_threshold, precursor_tol_mass,
                 precursor_tol_mode, pbar) for i in range(len(splits) - 1))
            max_label = _assign_global_cluster_labels(
                cluster_labels, idx, splits, max_label) + 1
    cluster_labels.flush()
    logger.info('%d embeddings grouped in %d clusters',
                (cluster_labels != -1).sum(), max_label)


@nb.njit
def _get_precursor_mz_splits(precursor_mzs: np.ndarray,
                             precursor_tol_mass: float,
                             precursor_tol_mode: str) -> nb.typed.List:
    """
    Find contiguous blocks of precursor m/z's, relative to the precursor m/z
    tolerance.

    Parameters
    ----------
    precursor_mzs : np.ndarray
        The sorted precursor m/z's.
    precursor_tol_mass : float
        The value of the precursor m/z tolerance.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').

    Returns
    -------
    nb.typed.List[int]
        A list of start and end indices of blocks of precursor m/z's that do
        not exceed the precursor m/z tolerance and are separated by at least
        the precursor m/z tolerance.
    """
    splits, i = nb.typed.List([0]), 1
    for i in range(1, len(precursor_mzs)):
        if suu.mass_diff(precursor_mzs[i], precursor_mzs[i - 1],
                         precursor_tol_mode == 'Da') > precursor_tol_mass:
            splits.append(i)
    splits.append(len(precursor_mzs))
    return splits


def _cluster_interval(embeddings: np.ndarray, idx: np.ndarray, mzs: np.ndarray,
                      cluster_labels: np.ndarray, interval_start: int,
                      interval_stop: int, linkage: str,
                      distance_threshold: float, precursor_tol_mass: float,
                      precursor_tol_mode: str, pbar: tqdm.tqdm) -> None:
    """
    Cluster the embeddings in the given interval.

    Parameters
    ----------
    embeddings : np.ndarray
        _All_ embeddings.
    idx : np.ndarray
        The indexes of the embeddings in the current interval.
    mzs : np.ndarray
        The precursor m/z's corresponding to the current interval indexes.
    cluster_labels : np.ndarray
        Array in which to fill the cluster label assignments.
    interval_start : int
        The current interval start index.
    interval_stop : int
        The current interval stop index.
    linkage : str
        Linkage method to calculate the cluster distances. See
        `scipy.cluster.hierarchy.linkage` for possible options.
    distance_threshold : float
        The maximum linkage distance threshold during clustering.
    precursor_tol_mass : float
        The value of the precursor m/z tolerance.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
    pbar : tqdm.tqdm
        Tqdm progress bar.
    """
    masks = np.arange(interval_start, interval_stop)
    np.random.shuffle(masks)
    masks = np.array_split(
        masks, math.ceil((interval_stop - interval_start) / 2 ** 15))
    for mask in masks:
        if len(mask) > 1:
            idx_interval, mzs_interval = idx[mask], mzs[mask]
            embeddings_interval = embeddings[idx_interval]
            # Hierarchical clustering of the embeddings.
            # Subtract 1 because fcluster starts with cluster label 1 instead
            # of 0 (like Scikit-Learn does).
            # with nb.objmode(labels='int32[:]'):
            pdist_euclidean = np.empty(len(mask) * (len(mask) - 1) // 2,
                                       np.float64)
            pdist(embeddings_interval, 'euclidean', out=pdist_euclidean)
            link_arr = np.empty((len(mask) - 1, 4), dtype=np.float64)
            fastcluster.linkage_wrap(len(mask), pdist_euclidean, link_arr,
                                     fastcluster.mthidx[linkage])
            del pdist_euclidean
            labels = fcluster(link_arr, distance_threshold, 'distance') - 1
            del link_arr
            if len(mask) > 2**10:
                gc.collect()
            # Refine initial clusters to make sure spectra within a cluster
            # don't have an excessive precursor m/z difference.
            order = np.argsort(labels)
            idx_interval = idx_interval[order]
            mzs_interval = mzs_interval[order]
            labels = labels[order]
            current_label = 0
            for start_i, stop_i in _get_cluster_group_idx(labels):
                n_clusters = _postprocess_cluster(
                    labels[start_i:stop_i], mzs_interval[start_i:stop_i],
                    precursor_tol_mass, precursor_tol_mode, 2, current_label)
                current_label += n_clusters
            # Assign cluster labels.
            cluster_labels[idx_interval] = labels
    pbar.update(interval_stop - interval_start)


@nb.njit(boundscheck=False)
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


@nb.njit(boundscheck=False)
def _postprocess_cluster(cluster_labels: np.ndarray, cluster_mzs: np.ndarray,
                         precursor_tol_mass: float, precursor_tol_mode: str,
                         min_samples: int, start_label: int) -> int:
    """
    Partitioning based on the precursor m/z's within each initial cluster to
    avoid that spectra within a cluster have an excessive precursor m/z
    difference.

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
    # No splitting needed if there are too few items in cluster.
    if cluster_labels.shape[0] < min_samples:
        cluster_labels.fill(-1)
        return 0
    else:
        # Group items within the cluster based on their precursor m/z.
        # Precursor m/z's within a single group can't exceed the specified
        # precursor m/z tolerance (`distance_threshold`).
        # Subtract 1 because fcluster starts with cluster label 1 instead of 0
        # (like Scikit-Learn does).
        linkage = _linkage(cluster_mzs, precursor_tol_mode)
        with nb.objmode(cluster_assignments='int32[:]'):
            cluster_assignments = fcluster(
                linkage, precursor_tol_mass, 'distance') - 1
        n_clusters = cluster_assignments.max() + 1
        # Update cluster assignments.
        if n_clusters == 1:
            # Single homogeneous cluster.
            cluster_labels.fill(start_label)
        elif n_clusters == cluster_mzs.shape[0]:
            # Only singletons.
            cluster_labels.fill(-1)
            n_clusters = 0
        else:
            labels = nb.typed.Dict.empty(key_type=nb.int64,
                                         value_type=nb.int64)
            for i, label in enumerate(cluster_assignments):
                labels[label] = labels.get(label, 0) + 1
            n_clusters = 0
            for label, count in labels.items():
                if count < min_samples:
                    labels[label] = -1
                else:
                    labels[label] = start_label + n_clusters
                    n_clusters += 1
            for i, label in enumerate(cluster_assignments):
                cluster_labels[i] = labels[label]
        return n_clusters


@nb.njit(fastmath=True, boundscheck=False)
def _linkage(mzs: np.ndarray, precursor_tol_mode: str) -> np.ndarray:
    """
    Linkage of a one-dimensional precursor m/z array.

    Because the data is one-dimensional, no pairwise distance matrix needs to
    be computed, but rather sorting can be used.

    For information on the linkage output format, see:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

    Parameters
    ----------
    mzs : np.ndarray
        The precursor m/z's for which pairwise distances are computed.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').

    Returns
    -------
    np.ndarray
        The hierarchical clustering encoded as a linkage matrix
    """
    linkage = np.zeros((mzs.shape[0] - 1, 4), np.double)
    # min m/z, max m/z, cluster index, number of cluster elements
    clusters = [(mzs[i], mzs[i], i, 1) for i in np.argsort(mzs)]
    for it in range(mzs.shape[0] - 1):
        min_dist, min_i = np.inf, -1
        for i in range(len(clusters) - 1):
            dist = suu.mass_diff(clusters[i + 1][1], clusters[i][0],
                                 precursor_tol_mode == 'Da')
            if dist < min_dist:
                min_dist, min_i = dist, i
        n_points = clusters[min_i][3] + clusters[min_i + 1][3]
        linkage[it, :] = [clusters[min_i][2], clusters[min_i + 1][2],
                          min_dist, n_points]
        clusters[min_i] = (clusters[min_i][0], clusters[min_i + 1][1],
                           mzs.shape[0] + it, n_points)
        del clusters[min_i + 1]

    return linkage


@nb.njit(fastmath=True, boundscheck=False)
def _assign_global_cluster_labels(cluster_labels: np.ndarray, idx: np.ndarray,
                                  splits: nb.typed.List, current_label: int) \
        -> int:
    """
    Convert cluster labels per split to globally unique labels.

    Parameters
    ----------
    cluster_labels : np.ndarray
        The cluster labels.
    idx : np.ndarray
        The label indexes.
    splits : nb.typed.List
        A list of start and end indices of cluster chunks.
    current_label : int
        First cluster label.

    Returns
    -------
    int
        Last cluster label.
    """
    max_label = current_label
    for i in range(len(splits) - 1):
        for j in idx[splits[i]:splits[i+1]]:
            if cluster_labels[j] != -1:
                cluster_labels[j] += current_label
                if cluster_labels[j] > max_label:
                    max_label = cluster_labels[j]
        current_label = max_label
    return max_label


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


@nb.njit(parallel=True, boundscheck=False)
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


@nb.njit(fastmath=True, boundscheck=False)
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
