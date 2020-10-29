import logging
import math
import os
from typing import List, Optional, Tuple

os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count())

import faiss
import joblib
import numba as nb
import numexpr as ne
import numpy as np
import pandas as pd
import scipy.sparse as ss
import tqdm
from sklearn.cluster import AgglomerativeClustering, DBSCAN
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
        config.num_neighbours = 1024


_check_ann_config()


def compute_pairwise_distances(embeddings_filename: str,
                               metadata_filename: str) -> None:
    """
    Compute a pairwise distance matrix for the embeddings in the given file.

    Parameters
    ----------
    embeddings_filename : str
        NumPy file containing the embedding vectors for which to compute
        pairwise distances.
    metadata_filename : str
        Metadata file with precursor m/z information for all embeddings.
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
    precursors = (pd.read_parquet(metadata_filename, columns=['charge', 'mz'])
                  .sort_values(['charge', 'mz']))
    precursors = precursors[precursors['charge'].isin(
        np.arange(config.charges[0], config.charges[1] + 1))]
    precursors = precursors.reset_index()
    embeddings = np.load(embeddings_filename, mmap_mode='r')
    embeddings = embeddings[precursors['index']]
    min_mz, max_mz = precursors['mz'].min(), precursors['mz'].max()
    mz_splits = np.arange(
        math.floor(min_mz / config.mz_interval) * config.mz_interval,
        math.ceil(max_mz / config.mz_interval) * config.mz_interval,
        config.mz_interval)
    # Create the ANN indexes (if this hasn't been done yet).
    _build_ann_index(index_filename, embeddings, precursors, mz_splits)
    # Calculate pairwise distances.
    num_embeddings = len(precursors)
    logging.info('Compute pairwise distances between neighboring embeddings '
                 '(%d embeddings, %d neighbors)', num_embeddings,
                 config.num_neighbors)
    if num_embeddings > np.iinfo(np.int32).max:
        raise OverflowError('Too many embedding indexes to fit into int32')
    if (not os.path.isfile(neighbors_filename.format('data')) or
            not os.path.isfile(neighbors_filename.format('indices')) or
            not os.path.isfile(neighbors_filename.format('indptr'))):
        distances, indices, indptr = [], [], [np.int32(0)]
        with tqdm.tqdm(total=precursors['charge'].nunique() * len(mz_splits),
                       desc='Distances calculated', unit='index') as pbar:
            for charge, precursors_charge in precursors.groupby('charge'):
                for mz in mz_splits:
                    _dist_mz_interval(
                        index_filename, embeddings, precursors_charge['mz'],
                        distances, indices, indptr, charge, mz)
                    pbar.update(1)
        np.save(neighbors_filename.format('data'),
                np.asarray(distances, np.float32))
        np.save(neighbors_filename.format('indices'),
                np.asarray(indices, np.int32))
        np.save(neighbors_filename.format('indptr'),
                np.asarray(indptr, np.int32))
    # Convert to a sparse pairwise distance matrix. This matrix might not be
    # entirely symmetrical, but that shouldn't matter too much.
    logger.debug('Construct pairwise distance matrix')
    pairwise_dist_matrix = ss.csr_matrix(
        ((np.load(neighbors_filename.format('data'), mmap_mode='r'),
          np.load(neighbors_filename.format('indices'), mmap_mode='r'),
          np.load(neighbors_filename.format('indptr'), mmap_mode='r'))),
        (num_embeddings, num_embeddings), np.float32, False)
    # Sort columns and rows by the original metadata/embeddings order.
    order = np.argsort(precursors['index'])
    pairwise_dist_matrix = pairwise_dist_matrix[order][:, order]
    logger.debug('Save the pairwise distance matrix to file %s', dist_filename)
    ss.save_npz(dist_filename, pairwise_dist_matrix, False)
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
                index.train(index_embeddings)
                # Add the embeddings to the index in batches.
                logger.debug('Add %d embeddings to the ANN index',
                             num_index_embeddings)
                batch_size = min(num_index_embeddings, config.batch_size_add)
                for batch_start in range(0, num_index_embeddings, batch_size):
                    batch_stop = min(batch_start + batch_size,
                                     num_index_embeddings)
                    index.add_with_ids(
                        index_embeddings[batch_start:batch_stop],
                        index_embeddings_ids[batch_start:batch_stop])
                # Save the index to disk.
                logger.debug('Save the ANN index to file %s',
                             index_filename.format(charge, mz))
                faiss.write_index(index, index_filename.format(charge, mz))
                index.reset()


def _dist_mz_interval(index_filename: str, embeddings: np.ndarray,
                      precursor_mzs: pd.Series, distances: List,
                      indices: List, indptr: List, charge: int, mz: int)\
        -> None:
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
    distances : List
        The nearest neighbor distances.
    indices : List
        The column indices for the nearest neighbor distances.
        See `scipy.sparse.csr_matrix`.
    indptr : List
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
    interval_ids = precursor_mzs.index.values[start_i:stop_i]
    for batch_start in range(0, len(interval_ids), config.batch_size_dist):
        batch_stop = batch_start + config.batch_size_dist
        batch_ids = interval_ids[batch_start:batch_stop]
        # Find nearest neighbors using ANN index searching.
        nn_dists, nn_idx_ann = index.search(
            embeddings[batch_ids], config.num_neighbors_ann)
        # Filter the neighbors based on the precursor m/z tolerance.
        nn_idx_mz = _get_neighbors_idx(
            precursor_mzs, precursor_mzs.loc[batch_ids].values)
        for idx_ann, idx_mz, dists in zip(nn_idx_ann, nn_idx_mz, nn_dists):
            mask = _intersect_idx_ann_mz(idx_ann, idx_mz,
                                         config.num_neighbors)
            distances.extend(dists[mask].astype(np.float32))
            indices.extend(idx_ann[mask].astype(np.int32))
            indptr.append(np.int32(indptr[-1] + len(mask)))
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
    idx = np.searchsorted(
        precursor_mzs, [start_mz - margin, start_mz + mz_window + margin])
    return idx[0], idx[1]


def _get_neighbors_idx(mzs: pd.Series, batch_mzs: np.ndarray) -> List:
    """
    Filter nearest neighbor candidates on precursor m/z.

    Parameters
    ----------
    mzs : pd.Series
        The precursor m/z's of the nearest neighbor candidates.
    batch_mzs : np.ndarray
        The precursor m/z's for which nearest neighbor candidates will be
        selected.

    Returns
    -------
    List
        A list of NumPy arrays with the indexes of the nearest neighbor
        candidates for each item.
    """
    precursor_tol_mass = config.precursor_tol_mass
    if config.precursor_tol_mode == 'Da':
        min_mz = batch_mzs[0] - precursor_tol_mass
        max_mz = batch_mzs[-1] + precursor_tol_mass
        mz_filter = 'abs(batch_mzs_arr - match_mzs_arr) < precursor_tol_mass'
    elif config.precursor_tol_mode == 'ppm':
        min_mz = batch_mzs[0] - batch_mzs[0] * precursor_tol_mass / 10**6
        max_mz = batch_mzs[-1] + batch_mzs[-1] * precursor_tol_mass / 10**6
        mz_filter = ('abs(batch_mzs_arr - match_mzs_arr)'
                     '/ match_mzs_arr * 10**6 < precursor_tol_mass')
    else:
        raise ValueError('Unknown precursor tolerance filter')
    batch_mzs_arr = batch_mzs.reshape((-1, 1))
    match_idx = np.searchsorted(mzs, [min_mz, max_mz])
    match_mzs_idx = mzs.index.values[match_idx[0]:match_idx[1]]
    match_mzs_arr = mzs.values[match_idx[0]:match_idx[1]].reshape((1, -1))
    return [match_mzs_idx[mask] for mask in ne.evaluate(mz_filter)]


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
        Identifiers from precursor m/z filtering.
    max_neighbors : int
        The maximum number of best matching neighbors to retain.

    Returns
    -------
    np.ndarray
        A mask to select the joint identifiers in the `idx_ann` array.
    """
    idx_mz, i_mz = np.sort(idx_mz), 0
    idx_ann_order = np.argsort(idx_ann)
    idx_ann_intersect = []
    for i_order, i_ann in enumerate(idx_ann_order):
        if idx_ann[i_ann] != -1:
            while i_mz < len(idx_mz) and idx_mz[i_mz] < idx_ann[i_ann]:
                i_mz += 1
            if i_mz == len(idx_mz):
                break
            if idx_ann[i_ann] == idx_mz[i_mz]:
                idx_ann_intersect.append(i_order)
                i_mz += 1
    # FIXME: Sorting could be avoided here using np.argpartition, but this is
    #        currently not supported by Numba.
    #        https://github.com/numba/numba/issues/2445
    return (np.sort(idx_ann_order[np.asarray(idx_ann_intersect)])
            [:max_neighbors])


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
    dbscan = DBSCAN(config.eps, min_samples=config.min_samples,
                    metric='precomputed', n_jobs=-1)
    cluster_labels = dbscan.fit_predict(ss.load_npz(distances_filename))
    logger.debug('Finetune %d initial cluster assignments to not exceed %d %s '
                 'precursor m/z tolerance', len(np.unique(cluster_labels)),
                 config.precursor_tol_mass, config.precursor_tol_mode)
    # Refine initial clusters to make sure spectra within a cluster don't have
    # an excessive precursor m/z difference..
    metadata = pd.read_parquet(metadata_filename, columns=['mz'])
    metadata['cluster'] = cluster_labels
    spectra_clusters = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(_postprocess_cluster)(spectra_cluster)
        for _, spectra_cluster in tqdm.tqdm(
            metadata[metadata['cluster'] != -1].groupby('cluster')['mz'],
            desc='Clusters post-processed', unit='cluster'))
    # Assign globally unique cluster labels.
    current_label = 0
    for spectra_cluster in spectra_clusters:
        spectra_cluster += current_label
        current_label += spectra_cluster.nunique()
    spectra_clusters = pd.concat(spectra_clusters)
    cluster_labels = -1 * np.ones_like(cluster_labels)
    cluster_labels[spectra_clusters.index] = spectra_clusters
    # Export the cluster assignments.
    logger.debug('%d embeddings partitioned in %d clusters',
                 len(cluster_labels), len(np.unique(cluster_labels)))
    logger.debug('Save the cluster assignments to file %s', clusters_filename)
    np.save(clusters_filename, cluster_labels)


def _postprocess_cluster(spectra_cluster: pd.Series) -> pd.Series:
    """
    Agglomerative clustering of the precursor m/z's within each initial
    cluster to avoid that spectra within a cluster have an excessive precursor
    m/z difference..

    Parameters
    ----------
    spectra_cluster : pd.Series
        A Series with precursor m/z values of the spectra that are grouped
        together during the initial clustering.

    Returns
    -------
    pd.Series
        A Series with cluster assignments starting at 0.
    """
    # No splitting possible if only 1 item in cluster.
    # This seems to happen sometimes despite that DBSCAN requires a higher
    # `min_samples`.
    if len(spectra_cluster) > 1:
        cluster_mzs = spectra_cluster.values.reshape(-1, 1)
        # Pairwise differences in Dalton.
        pairwise_mz_diff = pairwise_distances(cluster_mzs)
        if config.precursor_tol_mode == 'ppm':
            pairwise_mz_diff = pairwise_mz_diff / cluster_mzs * 10**6
        # Group items within the cluster based on their precursor m/z.
        # Precursor m/z's within a single group can't exceed the specified
        # precursor m/z tolerance (`distance_threshold`).
        clusterer = AgglomerativeClustering(
            n_clusters=None, affinity='precomputed', linkage='complete',
            distance_threshold=config.precursor_tol_mass)
        cluster_assignments = clusterer.fit_predict(pairwise_mz_diff)
        # Update cluster assignments.
        if clusterer.n_clusters_ == 1:
            return pd.Series(0, spectra_cluster.index)
        else:
            cluster_assignments_new = np.zeros_like(cluster_assignments)
            cluster_assignments = cluster_assignments.reshape(1, -1)
            labels = np.arange(1, clusterer.n_clusters_).reshape(-1, 1)
            for label, mask in zip(labels, cluster_assignments == labels):
                cluster_assignments_new[mask] = label
            return pd.Series(cluster_assignments_new, spectra_cluster.index)
    else:
        return pd.Series(0, spectra_cluster.index)
