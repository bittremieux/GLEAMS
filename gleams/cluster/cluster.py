import logging
import math
import os

import faiss
import numpy as np
import tqdm
from sklearn.cluster import DBSCAN
from scipy import sparse

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


def build_ann_index(embeddings_filename: str) -> None:
    """
    Create an ANN index for the given embedding vectors.

    If the number of embeddings exceeds the maximum ANN size specified in the
    config, the ANN index will be split over multiple sub-indexes.

    Parameters
    ----------
    embeddings_filename : str
        NumPy file containing the embedding vectors to build the ANN index.
    """
    ann_dir = os.path.join(os.environ['GLEAMS_HOME'], 'data', 'ann')
    if not os.path.isdir(ann_dir):
        os.makedirs(ann_dir)
    index_filename = os.path.splitext(
        os.path.basename(embeddings_filename))[0].replace('embed_', 'ann_')
    index_filename = os.path.join(ann_dir, f'{index_filename}.faiss')
    if os.path.isfile(index_filename):
        return
    # Create an ANN index using Euclidean distance for fast NN queries.
    embeddings = np.load(embeddings_filename, mmap_mode='r')
    num_embeddings = embeddings.shape[0]
    # Figure out a decent value for the num_list hyperparameter based on the
    # number of embeddings. Rules of thumb from the Faiss wiki:
    # https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index#how-big-is-the-dataset
    if num_embeddings < 10e5:
        num_list = 2**math.floor(math.log2(num_embeddings / 39))
    elif num_embeddings < 10e6:
        num_list = 2**16
    elif num_embeddings < 10e7:
        num_list = 2**18
    else:
        num_list = 2**20
        if num_embeddings > 10e8:
            logger.warning('More than 1B embeddings to be indexed, consider '
                           'decreasing the ANN size')
    logger.info('Build the ANN index using %d GPUs (%d embeddings, %d lists)',
                faiss.get_num_gpus(), num_embeddings, num_list)
    if not os.path.isfile(index_filename.replace('.faiss', '.coarse.faiss')):
        # Large datasets won't fit in the GPU memory, so we first compute
        # coarse cluster centroids using a subset of the data on the GPU.
        num_samples = min(num_embeddings, int(max(10e5, 256 * num_list)))
        logger.debug('Approximate the ANN index cluster centroids using %d '
                     'embeddings', num_samples)
        embeddings_sample = embeddings[np.random.choice(
            embeddings.shape[0], num_samples, False)]
        coarse_quantizer = _build_quantizer(embeddings_sample, num_list)
        index_cpu = faiss.IndexIVFFlat(
            coarse_quantizer, config.embedding_size, num_list, faiss.METRIC_L2)
        coarse_quantizer.this.disown()
        index_cpu.own_fields = True
        # Finish training on the CPU (CPU is batched automatically).
        index_cpu.train(embeddings)
        faiss.write_index(
            index_cpu, index_filename.replace('.faiss', '.coarse.faiss'))
    else:
        logger.debug('Load the cluster centroids from file %s', index_filename)
        index_cpu = faiss.read_index(
            index_filename.replace('.faiss', '.coarse.faiss'))
    # Add the embeddings to the index using the GPU for increased  performance.
    # Shard the GPU index over all available GPUs.
    logger.debug('Add the embeddings to the ANN index')
    # https://github.com/facebookresearch/faiss/blob/2cce2e5f59a5047aa9a1729141e773da9bec6b78/benchs/bench_gpu_1bn.py#L506
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    co.useFloat16 = True
    co.useFloat16CoarseQuantizer = False
    co.usePrecomputed = False
    co.indicesOptions = faiss.INDICES_CPU
    co.reserveVecs = config.ann_max_add
    index_gpu = faiss.index_cpu_to_all_gpus(index_cpu, co)
    # Add the embeddings in batches to avoid exhausting the GPU memory.
    batch_size = config.ann_add_batch_size
    for batch_start in tqdm.tqdm(range(0, num_embeddings, batch_size),
                                 desc='Batches processed', leave=False,
                                 unit='batch'):
        batch_stop = min(batch_start + batch_size, num_embeddings)
        index_gpu.add_with_ids(embeddings[batch_start:batch_stop],
                               np.arange(batch_start, batch_stop))
        if index_gpu.ntotal > config.ann_max_add:
            logger.debug('Flush ANN indexes to CPU after %d embeddings have '
                         'been added', batch_stop)
            if hasattr(index_gpu, 'at'):    # Sharded index.
                for i in range(index_gpu.count()):
                    index_src_gpu = faiss.downcast_index(index_gpu.at(i))
                    index_src = faiss.index_gpu_to_cpu(index_src_gpu)
                    index_src.copy_subset_to(index_cpu, 0, 0, num_embeddings)
                    index_src_gpu.reset()
                    index_src_gpu.reserveMemory(config.ann_max_add)
                index_gpu.sync_with_shard_indexes()
            else:       # Standard index.
                index_src = faiss.index_gpu_to_cpu(index_gpu)
                index_src.copy_subset_to(index_cpu, 0, 0, num_embeddings)
                index_gpu.reset()
                index_gpu.reserveMemory(config.ann_max_add)
    # Combine the sharded index into a single index and save.
    logger.debug('Save the ANN index to file %s', index_filename)
    # https://github.com/facebookresearch/faiss/blob/2cce2e5f59a5047aa9a1729141e773da9bec6b78/benchs/bench_gpu_1bn.py#L544
    if hasattr(index_gpu, 'at'):    # Sharded index.
        for i in range(index_gpu.count()):
            index_src = faiss.index_gpu_to_cpu(index_gpu.at(i))
            index_src.copy_subset_to(index_cpu, 0, 0, num_embeddings)
            index_gpu.at(i).reset()
    else:       # Standard index.
        index_src = faiss.index_gpu_to_cpu(index_gpu)
        index_src.copy_subset_to(index_cpu, 0, 0, num_embeddings)
        index_gpu.reset()
    faiss.write_index(index_cpu, index_filename)
    index_cpu.reset()
    if os.path.isfile(index_filename.replace('.faiss', '.coarse.faiss')):
        os.remove(index_filename.replace('.faiss', '.coarse.faiss'))


def _build_quantizer(x: np.ndarray, num_centroids: int) -> faiss.IndexFlatL2:
    """
    Build a quantizer with cluster centroids for ANN indexing using the
    Euclidean distance.

    The quantizer can be constructed using a subset of all data points,
    allowing it to be built using the GPU(s).

    Parameters
    ----------
    x : np.ndarray
        The data used to determine cluster centroids.
    num_centroids : int
        The number of centroids used by the quantizer.

    Returns
    -------
    faiss.IndexFlatL2
        The Faiss index to be used as quantizer.
    """
    # https://github.com/facebookresearch/faiss/blob/2cce2e5f59a5047aa9a1729141e773da9bec6b78/benchs/bench_gpu_1bn.py#L424
    clus = faiss.Clustering(config.embedding_size, num_centroids)
    clus.max_points_per_centroid = 10000000
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    co.useFloat16CoarseQuantizer = False
    co.usePrecomputed = False
    co.reserveVecs = x.shape[0]
    clus.train(x, faiss.index_cpu_to_all_gpus(
        faiss.IndexFlatL2(config.embedding_size), co))
    centroids = faiss.vector_float_to_array(clus.centroids)
    quantizer = faiss.IndexFlatL2(config.embedding_size)
    quantizer.add(centroids.reshape(num_centroids, config.embedding_size))

    return quantizer


def compute_pairwise_distances(embeddings_filename: str, ann_filename: str)\
        -> None:
    """
    Compute a pairwise distance matrix for the embeddings in the given file.
    The given ANN index should correspond to these embeddings as well and is
    used to perform efficient nearest neighbor queries.

    Parameters
    ----------
    embeddings_filename : str
        NumPy file containing the embedding vectors for which to compute
        pairwise distances.
    ann_filename : str
        Faiss index to perform nearest neighbor queries.
    """
    ann_dir = os.path.join(os.environ['GLEAMS_HOME'], 'data', 'ann')
    dist_filename = (os.path.splitext(
        os.path.basename(ann_filename))[0].replace('ann_', 'dist_'))
    dist_filename = os.path.join(ann_dir, f'{dist_filename}.npz')
    if os.path.isfile(dist_filename):
        return
    embeddings = np.load(embeddings_filename, mmap_mode='r')
    num_embeddings = embeddings.shape[0]
    logging.info('Compute pairwise distances between neighboring embeddings '
                 '(%d embeddings, %d neighbors)', num_embeddings,
                 config.num_neighbors)
    index = _load_ann_index(ann_filename)
    batch_size_query = min(num_embeddings, config.ann_search_batch_size)
    distances = np.empty(num_embeddings * config.num_neighbors, np.float32)
    neighbors = np.empty((num_embeddings * config.num_neighbors, 2), np.int64)
    neighbors[:, 0] = np.repeat(np.arange(num_embeddings, dtype=np.int64),
                                config.num_neighbors)
    for batch_i in tqdm.tqdm(range(0, num_embeddings, batch_size_query),
                             desc='Batches processed', leave=False,
                             unit='batch'):
        batch_start = batch_i
        batch_stop = min(batch_i + batch_size_query, num_embeddings)
        # Use the ANN index to efficiently compute the distances to the
        # nearest neighbors for the current batch of embeddings.
        dist_start = batch_start * config.num_neighbors
        dist_stop = (dist_start
                     + config.num_neighbors * (batch_stop - batch_start))
        batch_scores, batch_neighbors = index.search(
            embeddings[batch_start:batch_stop], config.num_neighbors)
        distances[dist_start:dist_stop] = batch_scores.flatten()
        neighbors[dist_start:dist_stop, 1] = batch_neighbors.flatten()
    index.reset()
    # Convert to a sparse pairwise distance matrix. This matrix might not be
    # entirely symmetrical, but that shouldn't matter too much.
    logger.debug('Construct pairwise distance matrix')
    idx = ~(neighbors[:, 1] == -1)
    neighbors, distances = neighbors[idx], distances[idx]
    pairwise_distances = sparse.csr_matrix(
        (distances, (neighbors[:, 0], neighbors[:, 1])),
        (num_embeddings, num_embeddings), np.float32, False)
    logger.debug('Save the pairwise distance matrix to file %s', dist_filename)
    sparse.save_npz(dist_filename, pairwise_distances, False)


def _load_ann_index(index_filename: str) -> faiss.Index:
    """
    Load the ANN index from the given file and move it to the GPU(s).

    Parameters
    ----------
    index_filename : str
        The ANN index filename.

    Returns
    -------
    faiss.Index
        The Faiss `Index`.
    """
    # https://github.com/facebookresearch/faiss/blob/2cce2e5f59a5047aa9a1729141e773da9bec6b78/benchs/bench_gpu_1bn.py#L608
    logger.debug('Load the ANN index from file %s', index_filename)
    index_cpu = faiss.read_index(index_filename)
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    co.useFloat16 = True
    co.useFloat16CoarseQuantizer = False
    co.usePrecomputed = False
    co.indicesOptions = faiss.INDICES_CPU
    co.reserveVecs = index_cpu.ntotal
    index = faiss.index_cpu_to_all_gpus(index_cpu, co)
    if hasattr(index, 'at'):
        for i in range(index.count()):
            faiss.downcast_index(index.at(i)).nprobe = config.num_probe
    else:
        index.nprobe = config.num_probe
    return index


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
    pairwise_distances = sparse.load_npz(distances_filename)
    clusters = dbscan.fit_predict(pairwise_distances)
    logger.debug('%d embeddings partitioned in %d clusters',
                 pairwise_distances.shape[0], len(np.unique(clusters)))
    logger.debug('Save the cluster assignments to file %s', clusters_filename)
    np.save(clusters_filename, clusters)
