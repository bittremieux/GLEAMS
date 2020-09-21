import logging
import os
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as ss
from tensorflow.keras import backend as K

from gleams import config
from gleams.feature import encoder, feature
from gleams.nn import data_generator, embedder


logger = logging.getLogger('gleams')


def _get_feature_split():
    return (config.num_precursor_features,
            config.num_precursor_features + config.num_fragment_features)


def train_nn(filename_model: str,
             filename_feat_train: str,
             filenames_train_pairs_pos: List[str],
             filenames_train_pairs_neg: List[str],
             filename_feat_val: str,
             filenames_val_pairs_pos: List[str],
             filenames_val_pairs_neg: List[str]):
    """
    Train the GLEAMS neural network.

    Parameters
    ----------
    filename_model : str
        The file name where the model will be saved.
    filename_feat_train : str
        The file name of the training SciPy sparse feature file.
    filenames_train_pairs_pos : List[str]
        The file names of the positive training pair indexes.
    filenames_train_pairs_neg : List[str]
        The file names of the negative training pair indexes.
    filename_feat_val : str
        The file name of the validation SciPy sparse feature file.
    filenames_val_pairs_pos : List[str]
        The file names of the positive validation pair indexes.
    filenames_val_pairs_neg : List[str]
        The file names of the negative validation pair indexes.
    """
    # Build the embedder model.
    model_dir = os.path.dirname(filename_model)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    logger.info('Compile the GLEAMS neural network')
    emb = embedder.Embedder(
        config.num_precursor_features, config.num_fragment_features,
        config.num_ref_spectra, config.lr, filename_model)
    emb.build()

    # Train the embedder.
    logger.info('Train the GLEAMS neural network')
    # Choose appropriate hyperparameters based on the number of GPUs that are
    # being used.
    if emb.num_gpu == 0:
        raise RuntimeError('No GPU found')
    batch_size = config.batch_size * emb.num_gpu
    steps_per_epoch = config.steps_per_epoch // emb.num_gpu
    if emb.num_gpu > 1:
        logger.info('Adjusting the batch size to %d and the steps per epoch to'
                    ' %d for running on %d GPUs', batch_size, steps_per_epoch,
                    emb.num_gpu)
    train_generator = data_generator.PairSequence(
        filename_feat_train, filenames_train_pairs_pos,
        filenames_train_pairs_neg, batch_size, _get_feature_split(),
        config.max_num_pairs_train)
    val_generator = data_generator.PairSequence(
        filename_feat_val, filenames_val_pairs_pos, filenames_val_pairs_neg,
        batch_size, _get_feature_split(), config.max_num_pairs_val, False)
    emb.train(train_generator, steps_per_epoch, config.num_epochs,
              val_generator)

    logger.info('Save the trained GLEAMS neural network')
    emb.save()

    logger.info('Training completed')


def embed(metadata_filename: str, model_filename: str,
          charges: Optional[Tuple[int]] = None) -> None:
    """
    Embed all spectra in the peak directory using the given GLEAMS model.

    Parameters
    ----------
    metadata_filename : str
        Metadata file with references to all datasets that should be embedded.
        Should be a Parquet file.
    model_filename : str
        The GLEAMS model filename.
    charges : Optional[Tuple[int]]
        Optional tuple of minimum and maximum precursor charge (both inclusive)
        to include, spectra with other precursor charges will be omitted.
    """
    embed_dir = os.path.join(os.environ['GLEAMS_HOME'], 'data', 'embed',
                             'dataset')
    if not os.path.isdir(embed_dir):
        os.makedirs(embed_dir)

    metadata = pd.read_parquet(
        metadata_filename, columns=['dataset', 'filename']).drop_duplicates()

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

    logger.debug('Load the stored GLEAMS neural network')
    emb = embedder.Embedder(
        config.num_precursor_features, config.num_fragment_features,
        config.num_ref_spectra, config.lr, model_filename)
    emb.load()
    if emb.num_gpu == 0:
        raise RuntimeError('No GPU found')
    batch_size = config.batch_size * emb.num_gpu

    logger.info('Embed all peak files for metadata file %s', metadata_filename)
    dataset_total = metadata['dataset'].nunique()
    for dataset_i, (dataset, peak_filenames) in enumerate(
            metadata.groupby('dataset', sort=False)['filename'], 1):
        filename_scans = os.path.join(embed_dir, f'{dataset}.parquet')
        filename_embedding = os.path.join(embed_dir, f'{dataset}.npy')
        if (os.path.isfile(filename_scans) and
                os.path.isfile(filename_embedding)):
            continue
        logger.info('Process dataset %s [%3d/%3d] (%d files)', dataset,
                    dataset_i, dataset_total, len(peak_filenames))
        peak_filenames_chunked = np.array_split(
            peak_filenames, max(1, len(peak_filenames) // 1000))
        scans = []
        for i, chunk_filenames in enumerate(peak_filenames_chunked):
            encodings = []
            for filename, file_scans, file_encodings in joblib.Parallel(
                    n_jobs=-1)(
                        joblib.delayed(feature._peaks_to_features)
                        (dataset, filename, None, enc)
                        for filename in chunk_filenames):
                if charges is not None and file_scans is not None:
                    file_scans = file_scans[
                        (file_scans['charge'] >= charges[0]) &
                        (file_scans['charge'] <= charges[1])].copy()
                if file_scans is not None and len(file_scans) > 0:
                    file_scans['dataset'] = dataset
                    file_scans['filename'] = filename
                    scans.append(file_scans)
                    encodings.extend(file_encodings[file_scans.index])
            if len(encodings) > 0:
                _embed_and_save(
                    encodings, batch_size, emb,
                    filename_embedding.replace('.npy', f'_{i}.npy'))
        if len(scans) > 0:
            scans = pd.concat(scans, ignore_index=True, sort=False, copy=False)
            scans[['dataset', 'filename', 'scan', 'charge', 'mz']].to_parquet(
                filename_scans, index=False)
            # Merge all temporary embeddings into a single file.
            embeddings = [np.load(filename_embedding.replace(
                              '.npy', f'_{i}.npy'), mmap_mode='r')
                          for i in range(len(peak_filenames_chunked))]
            np.save(filename_embedding, np.vstack(embeddings))
            for i in range(len(peak_filenames_chunked)):
                os.remove(filename_embedding.replace('.npy', f'_{i}.npy'))


def _embed_and_save(encodings: List[ss.csr_matrix], batch_size: int,
                    emb: embedder.Embedder, filename: str) -> None:
    """
    Embed the given encodings and save them as a NumPy file.

    Parameters
    ----------
    encodings : List[ss.csr_matrix]
        A list of encoding arrays to be embedded.
    batch_size : int
        The number of encodings to embed simultaneously.
    emb : embedder.Embedder
        The GLEAMS embedder.
    filename : str
        File name to store the embedded encodings.
    """
    logger.debug('Embed the spectrum encodings and save to file %s', filename)
    encodings_generator = data_generator.EncodingsSequence(
        ss.vstack(encodings, 'csr'), batch_size, _get_feature_split())
    np.save(filename, np.vstack(emb.embed(encodings_generator)))
    # FIXME: Avoid Keras memory leak.
    #        Possible issue: https://github.com/keras-team/keras/issues/13118
    K.clear_session()


def combine_embeddings(metadata_filename: str) -> None:
    """
    Combine embedding files for multiple datasets into a single embedding file.

    If the combined embedding file already exists it will _not_ be recreated.

    Parameters
    ----------
    metadata_filename : str
        Embeddings for all datasets included in the metadata will be combined.
        Should be a Parquet file.
    """
    embed_dir = os.path.join(os.environ['GLEAMS_HOME'], 'data', 'embed')
    embed_filename = os.path.join(embed_dir, os.path.splitext(
        os.path.basename(metadata_filename))[0].replace('metadata_', 'embed_'))
    if (os.path.isfile(f'{embed_filename}.npy') and
            os.path.isfile(f'{embed_filename}.parquet')):
        return
    datasets = pd.read_parquet(
        metadata_filename, columns=['dataset'])['dataset'].unique()
    logger.info('Combine embeddings for metadata file %s containing %d '
                'datasets', metadata_filename, len(datasets))
    embeddings, indexes = [], []
    for i, dataset in enumerate(datasets, 1):
        logger.debug('Append dataset %s [%3d/%3d]', dataset, i, len(datasets))
        dataset_embeddings_filename = os.path.join(
            embed_dir, 'dataset', f'{dataset}.npy')
        dataset_index_filename = os.path.join(
            embed_dir, 'dataset', f'{dataset}.parquet')
        if (not os.path.isfile(dataset_embeddings_filename) or
                not os.path.isfile(dataset_index_filename)):
            logger.warning('Missing embeddings for dataset %s, skipping...',
                           dataset)
        else:
            embeddings.append(np.load(dataset_embeddings_filename))
            indexes.append(pq.read_table(dataset_index_filename))
    np.save(f'{embed_filename}.npy', np.vstack(embeddings))
    pq.write_table(pa.concat_tables(indexes), f'{embed_filename}.parquet')
