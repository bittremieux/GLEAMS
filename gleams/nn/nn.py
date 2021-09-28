import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as ss
from tensorflow.keras import backend as K

from gleams.feature import encoder, feature
from gleams.nn import data_generator, embedder


logger = logging.getLogger('gleams')


def train_nn(filename_model: str,
             filename_feat_train: str,
             filenames_train_pairs_pos: List[str],
             filenames_train_pairs_neg: List[str],
             filename_feat_val: str,
             filenames_val_pairs_pos: List[str],
             filenames_val_pairs_neg: List[str],
             embedder_config: Dict[str, Any],
             batch_size: int,
             num_epochs: int,
             steps_per_epoch: int,
             max_num_pairs_train: int,
             max_num_pairs_val: int):
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
    embedder_config : Dict[str, Any]
        Configuration to initialize the embedder.
    batch_size : int
        The NN batch size.
    num_epochs : int
        The number of epochs to train for.
    steps_per_epoch : int
        The number of steps in a single epoch.
    max_num_pairs_train : int
        Maximum number of training pairs to include per combination of positive
        and negative file names.
    max_num_pairs_val : int
        Maximum number of validation pairs to include per combination of
        positive and negative file names.
    """
    # Build the embedder model.
    model_dir = os.path.dirname(filename_model)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    logger.info('Compile the GLEAMS neural network')
    emb = embedder.Embedder(filename=filename_model, **embedder_config)
    emb.build()

    # Train the embedder.
    logger.info('Train the GLEAMS neural network')
    # Choose appropriate hyperparameters based on the number of GPUs that are
    # being used.
    if emb.num_gpu == 0:
        raise RuntimeError('No GPU found')
    batch_size = batch_size * emb.num_gpu
    steps_per_epoch = steps_per_epoch // emb.num_gpu
    if emb.num_gpu > 1:
        logger.info('Adjusting the batch size to %d and the steps per epoch to'
                    ' %d for running on %d GPUs', batch_size, steps_per_epoch,
                    emb.num_gpu)
    feature_split = (embedder_config['num_precursor_features'],
                     embedder_config['num_precursor_features'] +
                     embedder_config['num_fragment_features'])
    train_generator = data_generator.PairSequence(
        filename_feat_train, filenames_train_pairs_pos,
        filenames_train_pairs_neg, batch_size, feature_split,
        max_num_pairs_train)
    validators = [
        data_generator.PairSequence(
            filename_feat_val, [filename_val_pairs_pos],
            [filename_val_pairs_neg], batch_size, feature_split,
            max_num_pairs_val, False)
        for filename_val_pairs_pos, filename_val_pairs_neg in zip(
            filenames_val_pairs_pos, filenames_val_pairs_neg)]
    emb.train(train_generator, steps_per_epoch, num_epochs, validators)

    logger.info('Save the trained GLEAMS neural network')
    emb.save()

    logger.info('Training completed')


def embed(metadata_filename: str,
          model_filename: str,
          embed_filename: str,
          embed_dir: str,
          precursor_encoding: Dict[str, Any],
          fragment_encoding: Dict[str, Any],
          reference_encoding: Dict[str, Any],
          embedder_config: Dict[str, Any],
          batch_size: int,
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
    embed_filename : str
        The embeddings file name to store the embedded spectra. Should have a
        ".npy" extension.
    embed_dir : str
        The local directory where the embedding files will be stored.
    precursor_encoding : Dict[str, Any]
        Settings for the precursor encoder.
    fragment_encoding : Dict[str, Any]
        Settings for the fragment encoder.
    reference_encoding : Dict[str, Any]
        Settings for the reference spectrum encoder.
    embedder_config : Dict[str, Any]
        Configuration to initialize the embedder.
    batch_size : int
        The NN batch size.
    charges : Optional[Tuple[int]]
        Optional tuple of minimum and maximum precursor charge (both inclusive)
        to include, spectra with other precursor charges will be omitted.
    """
    if not os.path.isdir(embed_dir):
        os.makedirs(embed_dir)

    metadata = pd.read_parquet(
        metadata_filename, columns=['dataset', 'filename']).drop_duplicates()

    enc = encoder.MultipleEncoder([
        encoder.PrecursorEncoder(**precursor_encoding),
        encoder.FragmentEncoder(**fragment_encoding),
        encoder.ReferenceSpectraEncoder(**reference_encoding)
    ])

    logger.debug('Load the stored GLEAMS neural network')
    emb = embedder.Embedder(filename=model_filename, **embedder_config)
    emb.load()
    if emb.num_gpu == 0:
        raise RuntimeError('No GPU found')
    batch_size = batch_size * emb.num_gpu
    feature_split = (embedder_config['num_precursor_features'],
                     embedder_config['num_precursor_features'] +
                     embedder_config['num_fragment_features'])

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
            peak_filenames, math.ceil(len(peak_filenames) / 200))
        scans = []
        for i, chunk_filenames in enumerate(peak_filenames_chunked):
            encodings = []
            # noinspection PyProtectedMember
            for filename, file_scans, file_encodings in joblib.Parallel(
                    n_jobs=-1)(
                        joblib.delayed(feature._peaks_to_features)
                        (dataset, filename, None, enc)
                        for filename in chunk_filenames):
                if file_scans is not None:
                    if charges is not None:
                        file_scans = file_scans[
                            (file_scans['charge'] >= charges[0]) &
                            (file_scans['charge'] <= charges[1])].copy()
                    if len(file_scans) > 0:
                        file_scans['dataset'] = dataset
                        file_scans['filename'] = filename
                        scans.append(file_scans)
                        encodings.extend(np.asarray(file_encodings)
                                         [file_scans.index.values])
            if len(encodings) > 0:
                _embed_and_save(
                    encodings, batch_size, feature_split, emb,
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
    # Combine all individual dataset embeddings.
    _combine_embeddings(
        embed_filename, embed_dir, metadata['dataset'].unique())


def _embed_and_save(encodings: List[ss.csr_matrix], batch_size: int,
                    feature_split: Tuple[int, int], emb: embedder.Embedder,
                    filename: str) -> None:
    """
    Embed the given encodings and save them as a NumPy file.

    Parameters
    ----------
    encodings : List[ss.csr_matrix]
        A list of encoding arrays to be embedded.
    batch_size : int
        The number of encodings to embed simultaneously.
    feature_split : Tuple[int, int]
        Indexes on which the feature vectors are split into individual inputs
        to the separate parts of the neural network (precursor features,
        fragment features, reference spectra features).
    emb : embedder.Embedder
        The GLEAMS embedder.
    filename : str
        File name to store the embedded encodings.
    """
    logger.debug('Embed the spectrum encodings and save to file %s', filename)
    encodings_generator = data_generator.EncodingsSequence(
        ss.vstack(encodings, 'csr'), batch_size, feature_split)
    np.save(filename, np.vstack(emb.embed(encodings_generator)))
    # FIXME: Avoid Keras memory leak.
    #        Possible issue: https://github.com/keras-team/keras/issues/13118
    K.clear_session()


def _combine_embeddings(filename: str, embed_dir: str, datasets: np.ndarray) \
        -> None:
    """
    Combine embedding files for multiple datasets into a single embedding file.

    If the combined embedding file already exists it will _not_ be recreated.

    Parameters
    ----------
    filename : str
        The embedding file name to store the embedded spectra. Should have a
        ".npy" extension.
    embed_dir : str
        Directory from which to read the embedding files for individual
        datasets.
    datasets : np.ndarray
        The datasets for which embedding files will be combined.
    """
    filename_embeddings = filename
    filename_index = f'{os.path.splitext(filename)[0]}.parquet'
    if os.path.isfile(filename_embeddings) and os.path.isfile(filename_index):
        return
    logger.info('Combine embeddings for %d datasets', len(datasets))
    embeddings, indexes = [], []
    for i, dataset in enumerate(datasets, 1):
        logger.debug('Append dataset %s [%3d/%3d]', dataset, i, len(datasets))
        dataset_embeddings_filename = os.path.join(embed_dir, f'{dataset}.npy')
        dataset_index_filename = os.path.join(embed_dir, f'{dataset}.parquet')
        if (not os.path.isfile(dataset_embeddings_filename) or
                not os.path.isfile(dataset_index_filename)):
            logger.warning('Missing embeddings for dataset %s, skipping...',
                           dataset)
        else:
            embeddings.append(np.load(dataset_embeddings_filename))
            indexes.append(pq.read_table(dataset_index_filename))
    np.save(filename_embeddings, np.vstack(embeddings))
    pq.write_table(pa.concat_tables(indexes), filename_index)
