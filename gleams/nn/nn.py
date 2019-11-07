import logging
import os

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gleams import config
from gleams.feature import encoder, feature
from gleams.nn import data_generator, embedder


logger = logging.getLogger('gleams')


def _get_feature_split():
    return (config.num_precursor_features,
            config.num_precursor_features + config.num_fragment_features)


def train_nn(filename_model: str, filename_feat_train: str,
             filename_train_pairs_pos: str, filename_train_pairs_neg: str,
             filename_feat_val: str, filename_val_pairs_pos: str,
             filename_val_pairs_neg: str):
    """
    Train the GLEAMS neural network.

    Parameters
    ----------
    filename_model : str
        The file name where the model will be saved.
    filename_feat_train : str
        The file name of the training NumPy binary feature file.
    filename_train_pairs_pos : str
        The file name of the positive training pair indexes.
    filename_train_pairs_neg : str
        The file name of the negative training pair indexes.
    filename_feat_val : str
        The file name of the validation NumPy binary feature file.
    filename_val_pairs_pos : str
        The file name of the positive validation pair indexes.
    filename_val_pairs_neg : str
        The file name of the negative validation pair indexes.
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
    num_gpus = embedder._get_num_gpus()
    if num_gpus == 0:
        raise RuntimeError('No GPU found')
    batch_size = config.batch_size * num_gpus
    steps_per_epoch = config.steps_per_epoch // num_gpus
    if num_gpus > 1:
        logger.info('Adjusting the batch size to %d and the steps per epoch to'
                    ' %d for running on %d GPUs', batch_size, steps_per_epoch,
                    num_gpus)
    train_generator = data_generator.PairSequence(
        filename_feat_train, filename_train_pairs_pos,
        filename_train_pairs_neg, batch_size, _get_feature_split(),
        config.max_num_pairs_train)
    val_generator = data_generator.PairSequence(
        filename_feat_val, filename_val_pairs_pos, filename_val_pairs_neg,
        batch_size, _get_feature_split(), config.max_num_pairs_val, False)
    emb.train(train_generator, steps_per_epoch, config.num_epochs,
              val_generator)

    logger.info('Save the trained GLEAMS neural network')
    emb.save()

    logger.info('Training completed')


def embed(metadata_filename: str, model_filename: str) -> None:
    """
    Embed all spectra in the peak directory using the given GLEAMS model.

    Parameters
    ----------
    metadata_filename : str
        Metadata file with references to all datasets that should be embedded.
    model_filename : str
        The GLEAMS model filename.
    """
    peak_dir = os.path.join(os.environ['GLEAMS_HOME'], 'data', 'peak')
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
    num_gpus = embedder._get_num_gpus()
    if num_gpus == 0:
        raise RuntimeError('No GPU found')
    batch_size = config.batch_size * num_gpus

    logger.info('Embed all spectra in directory %s', peak_dir)
    dataset_total = metadata['dataset'].nunique()
    for dataset_i, (dataset, peak_filenames) in enumerate(
            metadata.groupby('dataset', sort=False)['filename'], 1):
        metadata_filename = os.path.join(embed_dir, f'{dataset}.parquet')
        filename_embedding = os.path.join(embed_dir, f'{dataset}.npy')
        if (os.path.isfile(metadata_filename) and
                os.path.isfile(filename_embedding)):
            continue
        logger.info('Process dataset %s [%3d/%3d] (%d files)', dataset,
                    dataset_i, dataset_total, len(peak_filenames))
        encodings, metadata = [], []
        for filename, file_scans, file_encodings in joblib.Parallel(n_jobs=-1)(
                joblib.delayed(feature._peaks_to_features)
                (dataset, filename, None, enc) for filename in peak_filenames):
            if file_scans is not None and len(file_scans) > 0:
                metadata.extend([(dataset, filename, scan)
                                 for scan in file_scans])
                encodings.extend(file_encodings)
        if len(metadata) > 0:
            pq.write_table(pa.Table.from_pandas(pd.DataFrame(
                metadata, columns=['dataset', 'filename', 'scan'])),
                metadata_filename)
            logger.debug('Embed the spectrum encodings')
            embeddings = emb.embed(
                list(data_generator._split_features_to_input(
                    encodings, *_get_feature_split())), batch_size)
            np.save(filename_embedding, np.vstack(embeddings))
