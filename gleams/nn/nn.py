import logging
import os

from gleams import config
from gleams.nn import data_generator, embedder


logger = logging.getLogger('gleams')


def train_nn(filename_feat: str, filename_model: str,
             filename_metadata_train: str,  filename_train_pairs_pos: str,
             filename_train_pairs_neg: str,
             filename_metadata_val: str, filename_val_pairs_pos: str,
             filename_val_pairs_neg: str):
    """
    Train the GLEAMS neural network.

    Parameters
    ----------
    filename_feat : str
        The file name of the HDF5 feature file.
    filename_model : str
        The file name where the model will be saved.
    filename_metadata_train : str
        The file name of the training PSM metadata file.
    filename_train_pairs_pos : str
        The file name of the positive training pair indexes.
    filename_train_pairs_neg : str
        The file name of the negative training pair indexes.
    filename_metadata_val : str
        The file name of the validation PSM metadata file.
    filename_val_pairs_pos : str
        The file name of the positive validation pair indexes.
    filename_val_pairs_neg : str
        The file name of the negative validation pair indexes.
    """
    # Build the embedder model.
    model_dir = os.path.dirname(filename_model)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    logger.info('Compile the GLEAMS siamese neural network')
    emb = embedder.Embedder(
        config.num_precursor_features, config.num_fragment_features,
        config.num_ref_spectra, config.lr, filename_model)
    emb.build_siamese_model()

    # Train the embedder.
    logger.info('Train the GLEAMS siamese neural network')
    feature_split = (config.num_precursor_features,
                     config.num_precursor_features + config.num_fragment_features)
    with data_generator.PairSequence(
                filename_metadata_train, filename_feat,
                filename_train_pairs_pos, filename_train_pairs_neg,
                config.batch_size, feature_split,
                config.max_num_pairs_train) as train_generator,\
            data_generator.PairSequence(
                filename_metadata_val, filename_feat,
                filename_val_pairs_pos, filename_val_pairs_neg,
                config.batch_size, feature_split,
                config.max_num_pairs_val, False) as val_generator:
        emb.train(train_generator, config.steps_per_epoch, config.num_epochs,
                  val_generator)

    logger.info('Save the trained GLEAMS siamese neural network')
    emb.save()

    logger.info('Training completed')
