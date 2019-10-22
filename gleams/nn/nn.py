import logging
import os

from keras.utils import multi_gpu_utils

from gleams import config
from gleams.nn import data_generator, embedder


logger = logging.getLogger('gleams')


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
    logger.info('Compile the GLEAMS siamese neural network')
    emb = embedder.Embedder(
        config.num_precursor_features, config.num_fragment_features,
        config.num_ref_spectra, config.lr, filename_model)
    emb.build_siamese_model()

    # Train the embedder.
    logger.info('Train the GLEAMS siamese neural network')
    feature_split =\
        (config.num_precursor_features,
         config.num_precursor_features + config.num_fragment_features)
    # Choose appropriate hyperparameters based on the number of GPUs that are
    # being used.
    available_devices = [
        multi_gpu_utils._normalize_device_name(name)
        for name in multi_gpu_utils._get_available_devices()]
    num_gpus = len([x for x in available_devices if '/gpu' in x])
    if num_gpus == 0:
        raise RuntimeError(f'No GPU found (available devices: '
                           f'{", ".join(available_devices)})')
    batch_size = config.batch_size * num_gpus
    steps_per_epoch = config.steps_per_epoch // num_gpus
    if num_gpus > 1:
        logger.info('Adjusting the batch size to %d and the steps per epoch to'
                    ' %d for running on %d GPUs', batch_size, steps_per_epoch,
                    num_gpus)
    train_generator = data_generator.PairSequence(
        filename_feat_train, filename_train_pairs_pos,
        filename_train_pairs_neg, batch_size, feature_split,
        config.max_num_pairs_train)
    val_generator = data_generator.PairSequence(
        filename_feat_val, filename_val_pairs_pos, filename_val_pairs_neg,
        batch_size, feature_split, config.max_num_pairs_val, False)
    emb.train(train_generator, steps_per_epoch, config.num_epochs,
              val_generator)

    logger.info('Save the trained GLEAMS siamese neural network')
    emb.save()

    logger.info('Training completed')
