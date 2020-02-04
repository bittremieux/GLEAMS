import logging
import os

os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count())

import numba as nb
import numexpr as ne
import numpy as np
import pandas as pd
import scipy.sparse as ss
import tqdm
from sklearn.cluster import DBSCAN

from gleams import config


logger = logging.getLogger('gleams')


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
    dist_filename = (os.path.splitext(
        os.path.basename(embeddings_filename))[0].replace('embed_', 'dist_'))
    dist_filename = os.path.join(cluster_dir, f'{dist_filename}.npz')
    neighbors_filename = (dist_filename.replace('dist_', 'neighbors_{}_')
                                       .replace('.npz', '.npy'))
    if os.path.isfile(dist_filename):
        return
    embeddings = np.load(embeddings_filename, mmap_mode='r')
    num_embeddings = embeddings.shape[0]
    precursor_mzs = (pd.read_parquet(metadata_filename, columns=['mz'])
                     .squeeze().sort_values())
    logging.info('Compute pairwise distances between neighboring embeddings '
                 '(%d embeddings, %d neighbors)', num_embeddings,
                 config.num_neighbors)
    if num_embeddings > np.iinfo(np.uint32).max:
        raise OverflowError('Too many embedding indexes to fit into uint32')
    neighbors = np.empty((num_embeddings * config.num_neighbors), np.uint32)
    distances = np.full(num_embeddings * config.num_neighbors, np.nan,
                        np.float32)
    batch_size = min(num_embeddings, config.dist_batch_size)
    for batch_i in tqdm.tqdm(range(0, num_embeddings, batch_size),
                             desc='Batches processed', leave=False,
                             unit='batch'):
        batch_start = batch_i
        batch_stop = min(batch_i + batch_size, num_embeddings)
        dist_start = batch_start * config.num_neighbors
        dist_stop = (dist_start
                     + config.num_neighbors * (batch_stop - batch_start))
        # Filter neighbors on precursor m/z.
        batch_mzs = (precursor_mzs.iloc[batch_start:batch_stop]
                     .values.reshape((-1, 1)))
        precursor_tol_mass = config.precursor_tol_mass
        if config.precursor_tol_mode == 'Da':
            batch_mzs_match = precursor_mzs[precursor_mzs.between(
                batch_mzs[0, 0] - 1.5 * precursor_tol_mass,
                batch_mzs[-1, 0] + 1.5 * precursor_tol_mass)]
            batch_mzs_match_arr = batch_mzs_match.values.reshape((1, -1))
            neighbors_masks = ne.evaluate(
                'abs(batch_mzs - batch_mzs_match_arr) <= precursor_tol_mass')
        elif config.precursor_tol_mode == 'ppm':
            batch_mzs_match = precursor_mzs[precursor_mzs.between(
                batch_mzs[0, 0] - 1.5 * batch_mzs[0, 0]
                                      * precursor_tol_mass / 10**6,
                batch_mzs[-1, 0] + 1.5 * batch_mzs[-1, 0]
                                       * precursor_tol_mass / 10**6)]
            batch_mzs_match_arr = batch_mzs_match.values.reshape((1, -1))
            neighbors_masks = ne.evaluate(
                'abs(batch_mzs - batch_mzs_match_arr)'
                '   / batch_mzs_match_arr * 10**6 <= precursor_tol_mass')
        else:
            raise ValueError('Unknown precursor tolerance filter')
        for embedding_i, neighbors_mask, dist_i in zip(
                precursor_mzs.index[batch_start:batch_stop], neighbors_masks,
                np.arange(dist_start, dist_stop, config.num_neighbors)):
            _set_nn(embeddings, neighbors, distances, embedding_i,
                    batch_mzs_match.index[np.where(neighbors_mask)[0]].values,
                    dist_i)
    mask = np.where(~np.isnan(distances))[0]
    np.save(neighbors_filename.format(1), neighbors[mask])
    np.save(neighbors_filename.format('distance'), distances[mask])
    np.save(neighbors_filename.format(0),
            np.repeat(np.asarray(precursor_mzs.index, dtype=np.uint32),
                      config.num_neighbors)[mask])
    # Convert to a sparse pairwise distance matrix. This matrix might not be
    # entirely symmetrical, but that shouldn't matter too much.
    logger.debug('Construct pairwise distance matrix')
    pairwise_distances = ss.csr_matrix(
        (np.load(neighbors_filename.format('distance'), mmap_mode='r'),
         (np.load(neighbors_filename.format(0), mmap_mode='r'),
          np.load(neighbors_filename.format(1), mmap_mode='r'))),
        (num_embeddings, num_embeddings), np.float32, False)
    logger.debug('Save the pairwise distance matrix to file %s', dist_filename)
    ss.save_npz(dist_filename, pairwise_distances, False)
    os.remove(neighbors_filename.format(0))
    os.remove(neighbors_filename.format(1))
    os.remove(neighbors_filename.format('distance'))


@nb.njit
def _set_nn(embeddings: np.ndarray, neighbors: np.ndarray,
            distances: np.ndarray, embedding_i: int, neighbors_i: np.ndarray,
            dist_i: int) -> None:
    """
    Assign nearest neighbor distances and indexes.

    Parameters
    ----------
    embeddings : np.ndarray
        Matrix containing the embeddings.
    neighbors : np.ndarray
        Vector with neighbor indexes to be assigned.
    distances : np.ndarray
        Vector with neighbor distances to be assigned.
    embedding_i : int
        The index of the embedding for which the nearest neighbors are
        evaluated.
    neighbors_i : np.ndarray
        The indexes of the nearest neighbors for the embedding to be evaluated.
    dist_i : int
        The start index of the embedding in the neighbor vectors.
    """
    # Upcast to avoid float32 precision errors.
    neighbors_dist = _cdist(
        embeddings[embedding_i].reshape(1, -1).astype(np.float64),
        embeddings[neighbors_i].astype(np.float64))[0]
    # Restrict to `num_neighbors` closest neighbors.
    if len(neighbors_i) > config.num_neighbors:
        neighbors_dist_mask = (np.argsort(neighbors_dist)
                               [:config.num_neighbors])
        neighbors_i = neighbors_i[neighbors_dist_mask]
        neighbors_dist = neighbors_dist[neighbors_dist_mask]
    neighbors[dist_i:dist_i + len(neighbors_i)] = neighbors_i
    distances[dist_i:dist_i + len(neighbors_i)] = neighbors_dist


@nb.njit
def _cdist(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute the Eucliedean distance between each pair of the two collections of
    inputs.

    Similar to https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    using Euclidean distance.

    Parameters
    ----------
    X : np.ndarray
        An $m_X$ by $n$ array of original observations in an $n$-dimensional
        space. Inputs should be float64 type.
    Y : np.ndarray
        An $m_Y$ by $n$ array of original observations in an $n$-dimensional
        space. Inputs should be float64 type.

    Returns
    -------
    np.ndarray
        A $m_X$ by $m_Y$ distance matrix is returned. For each $i$ and $j$, the
        Euclidean distance `dist(u=XA[i], v=XB[j])` is computed and stored in
        the $ij$th entry.
    """
    XX, YY = (X ** 2).sum(axis=1), (Y ** 2).sum(axis=1)
    dist = - 2 * np.dot(X, Y.T) + XX + YY
    return np.sqrt(np.maximum(dist, 0, dist))


def cluster(distances_filename: str):
    """
    DBSCAN clustering of the embeddings based on a pairwise distance matrix.

    Parameters
    ----------
    distances_filename : str
        Precomputed pairwise distance matrix file to use for the DBSCAN
        clustering.
    """
    clusters_filename = (distances_filename.replace('dist_', 'clusters_')
                                           .replace('.npz', '.npy'))
    if os.path.isfile(clusters_filename):
        return
    logger.info('DBSCAN clustering (eps=%.4f, min_samples=%d) of precomputed '
                'pairwise distance matrix %s', config.eps, config.min_samples,
                distances_filename)
    dbscan = DBSCAN(config.eps, config.min_samples, 'precomputed', n_jobs=-1)
    pairwise_distances = ss.load_npz(distances_filename)
    clusters = dbscan.fit_predict(pairwise_distances)
    logger.debug('%d embeddings partitioned in %d clusters',
                 pairwise_distances.shape[0], len(np.unique(clusters)))
    logger.debug('Save the cluster assignments to file %s', clusters_filename)
    np.save(clusters_filename, clusters)
