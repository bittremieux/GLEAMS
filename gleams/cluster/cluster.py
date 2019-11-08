import logging
import math
import os
from typing import Dict, List

import faiss
import joblib
import numpy as np
import pandas as pd
import tqdm
from sklearn.cluster import DBSCAN
from sklearn.metrics import homogeneity_completeness_v_measure
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
    if config.num_neighbours > 1024:
        logger.warning('Using num_neighbours=1024 (maximum supported value '
                       'for GPU-enabled ANN indexing), %d was supplied',
                       config.num_neighbours)
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
    # Create an ANN index using Euclidean distance for fast NN queries.
    embeddings = np.load(embeddings_filename, mmap_mode='r')
    num_embeddings = embeddings.shape[0]
    # Figure out a decent value for the num_list hyperparameter based on the
    # number of embeddings. Rules of thumb from the Faiss wiki:
    # https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index#how-big-is-the-dataset
    if num_embeddings < 10e6:
        num_list = 2**math.floor(math.log2(num_embeddings / 39))
    elif num_embeddings < 10e7:
        num_list = 2**16
    elif num_embeddings < 10e8:
        num_list = 2**18
    else:
        num_list = 2**20
        if num_embeddings > 10e9:
            logger.warning('More than 1B vectors to be indexed at the same '
                           'time, consider decreasing the ANN size')
    logger.info(f'Build the ANN index (%d embeddings, %d lists)',
                num_embeddings, num_list)
    # Training the ANN index needs to happen on the CPU because it can't
    # be batched.
    # TODO: Prepare the quantizer.
    #   https://github.com/facebookresearch/faiss/blob/2cce2e5f59a5047aa9a1729141e773da9bec6b78/benchs/bench_gpu_1bn.py#L424
    index_cpu = faiss.IndexIVFFlat(
        faiss.IndexFlatL2(config.embedding_size), config.embedding_size,
        num_list, faiss.METRIC_L2)
    logger.debug('Train the ANN index on the CPU')
    index_cpu.train(embeddings)
    # Add the embedding vectors to the index using the GPU for increased
    # performance.
    # Shard the GPU index over all available GPUs.
    num_gpus = faiss.get_num_gpus()
    logger.debug('Add the embeddings to the ANN index using %d GPU(s)',
                 num_gpus)
    # https://github.com/facebookresearch/faiss/blob/2cce2e5f59a5047aa9a1729141e773da9bec6b78/benchs/bench_gpu_1bn.py#L506
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    co.useFloat16 = True
    co.useFloat16CoarseQuantizer = False
    co.indicesOptions = faiss.INDICES_CPU
    co.reserveVecs = num_embeddings
    vres, vdev = faiss.GpuResourcesVector(), faiss.IntVector()
    for i in range(num_gpus):
        vres.push_back(faiss.StandardGpuResources())
        vdev.push_back(i)
    index_gpu = faiss.index_cpu_to_all_gpus(vres, vdev, index_cpu, co)
    # Add the embeddings in batches to avoid exhausting the GPU memory.
    batch_size = config.ann_add_batch_size
    for batch_start in tqdm.tqdm(range(0, num_embeddings, batch_size),
                                 desc='Batches processed', leave=False,
                                 unit='batch'):
        batch_stop = min(batch_start + batch_size, num_embeddings)
        index_gpu.add_with_ids(embeddings[batch_start:batch_stop],
                               np.arange(batch_start, batch_stop))
    # Combine the sharded index into a single index and save.
    logger.debug('Save the ANN index')
    # https://github.com/facebookresearch/faiss/blob/2cce2e5f59a5047aa9a1729141e773da9bec6b78/benchs/bench_gpu_1bn.py#L544
    if hasattr(index_gpu, 'at'):    # Sharded index.
        for i in range(num_gpus):
            index_src = faiss.index_gpu_to_cpu(index_gpu.at(i))
            index_src.copy_subset_to(index_cpu, 0, 0, num_embeddings)
            index_gpu.at(i).reset()
    else:       # Standard index.
        index_src = faiss.index_gpu_to_cpu(index_gpu)
        index_src.copy_subset_to(index_cpu, 0, 0, num_embeddings)
    faiss.write_index(index_cpu, os.path.join(
        os.environ['GLEAMS_HOME'], 'data', 'ann', f'ann_index.faiss'))
    index_cpu.reset()
