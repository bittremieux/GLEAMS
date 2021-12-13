import gc
import logging
import math
import os
import warnings
from typing import Iterator, List, Optional, Tuple

import fastcluster
import joblib
import numba as nb
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import spectrum_utils.utils as suu
import tqdm


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
        File name to export the cluster labels. Must have a ".npy" extension.
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
    if clusters_dir and not os.path.exists(clusters_dir):
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
    max_label, medoids = 0, []
    with tqdm.tqdm(total=len(metadata), desc='Clustering', unit='embedding',
                   smoothing=0) as pbar:
        for _, metadata_charge in metadata.groupby('charge'):
            idx = metadata_charge['index'].values
            mz = metadata_charge['mz'].values
            splits = _get_precursor_mz_splits(mz, precursor_tol_mass,
                                              precursor_tol_mode, 2**15)
            # Per-split cluster labels.
            for interval_medoids in joblib.Parallel(
                    n_jobs=-1, backend='threading')(
                joblib.delayed(_cluster_interval)
                    (embeddings, idx, mz, cluster_labels, splits[i],
                     splits[i+1], linkage, distance_threshold,
                     precursor_tol_mass, precursor_tol_mode, pbar)
                    for i in range(len(splits) - 1)):
                if interval_medoids is not None:
                    medoids.append(interval_medoids)
            max_label = _assign_global_cluster_labels(
                cluster_labels, idx, splits, max_label)
    cluster_labels.flush()
    np.save(clusters_filename.replace('.npy', '_medoids.npy'),
            np.hstack(medoids))
    logger.info('%d embeddings grouped in %d clusters',
                (cluster_labels != -1).sum(), max_label)


@nb.njit
def _get_precursor_mz_splits(precursor_mzs: np.ndarray,
                             precursor_tol_mass: float,
                             precursor_tol_mode: str,
                             batch_size: int) -> nb.typed.List:
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
    batch_size : int
        Maximum interval size.

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
            block_size = i - splits[-1]
            if block_size < batch_size:
                splits.append(i)
            else:
                n_chunks = math.ceil(block_size / batch_size)
                chunk_size = block_size // n_chunks
                for _ in range(block_size % n_chunks):
                    splits.append(splits[-1] + chunk_size + 1)
                for _ in range(n_chunks - (block_size % n_chunks)):
                    splits.append(splits[-1] + chunk_size)
    splits.append(len(precursor_mzs))
    return splits


def _cluster_interval(embeddings: np.ndarray, idx: np.ndarray, mzs: np.ndarray,
                      cluster_labels: np.ndarray, interval_start: int,
                      interval_stop: int, linkage: str,
                      distance_threshold: float, precursor_tol_mass: float,
                      precursor_tol_mode: str, pbar: tqdm.tqdm) \
        -> Optional[np.ndarray]:
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

    Returns
    -------
    Optional[np.ndarray]
        List with indexes of the medoids for each cluster.
    """
    n_embeddings = interval_stop - interval_start
    if n_embeddings > 1:
        idx_interval = idx[interval_start:interval_stop]
        mzs_interval = mzs[interval_start:interval_stop]
        embeddings_interval = embeddings[idx_interval]
        # Hierarchical clustering of the embeddings.
        # Subtract 1 because fcluster starts with cluster label 1 instead of 0
        # (like Scikit-Learn does).
        pdist = np.empty(n_embeddings * (n_embeddings - 1) // 2, np.float64)
        ssd.pdist(embeddings_interval, 'euclidean', out=pdist)
        labels = sch.fcluster(fastcluster.linkage(pdist, linkage),
                              distance_threshold, 'distance') - 1
        # Refine initial clusters to make sure spectra within a cluster don't
        # have an excessive precursor m/z difference.
        order = np.argsort(labels)
        idx_interval, mzs_interval = idx_interval[order], mzs_interval[order]
        labels, current_label = labels[order], 0
        for start_i, stop_i in _get_cluster_group_idx(labels):
            n_clusters = _postprocess_cluster(
                labels[start_i:stop_i], mzs_interval[start_i:stop_i],
                precursor_tol_mass, precursor_tol_mode, 2, current_label)
            current_label += n_clusters
        # Assign cluster labels.
        cluster_labels[idx_interval] = labels
        if current_label > 0:
            # Compute cluster medoids.
            order_ = np.argsort(labels)
            idx_interval, labels = idx_interval[order_], labels[order_]
            order_map = order[order_]
            medoids = _get_cluster_medoids(idx_interval, labels, pdist,
                                           order_map)
        else:
            medoids = None
        # Force memory clearing.
        del pdist
        if n_embeddings > 2**11:
            gc.collect()
    else:
        medoids = None
    pbar.update(n_embeddings)
    return medoids


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
            cluster_assignments = sch.fcluster(
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
def _get_cluster_medoids(idx_interval: np.ndarray, labels: np.ndarray,
                         pdist: np.ndarray, order_map: np.ndarray) \
        -> np.ndarray:
    """
    Get the indexes of the cluster medoids.

    Parameters
    ----------
    idx_interval : np.ndarray
        Embedding indexes.
    labels : np.ndarray
        Cluster labels.
    pdist : np.ndarray
        Condensed pairwise distance matrix.
    order_map : np.ndarray
        Map to convert label indexes to pairwise distance matrix indexes.

    Returns
    -------
    List[int]
        List with indexes of the medoids for each cluster.
    """
    medoids, m = [], len(idx_interval)
    for start_i, stop_i in _get_cluster_group_idx(labels):
        if stop_i - start_i > 1:
            row_sum = np.zeros(stop_i - start_i, np.float32)
            for row in range(stop_i - start_i):
                for col in range(row + 1, stop_i - start_i):
                    i, j = order_map[start_i + row], order_map[start_i + col]
                    if i > j:
                        i, j = j, i
                    pdist_ij = pdist[m * i + j - ((i + 2) * (i + 1)) // 2]
                    row_sum[row] += pdist_ij
                    row_sum[col] += pdist_ij
            medoids.append(idx_interval[start_i + np.argmin(row_sum)])
    return np.asarray(medoids, dtype=np.int32)


@nb.njit(boundscheck=False)
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
        current_label = max_label + 1
    return max_label
