# Limit annoying Tensforflow logging to only warnings and errors.
import os
# 1: No FILTER logging.
# 2: No WARNING logging.
# 3: No ERROR logging.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from gleams.embed import rndm
rndm.set_seeds()

import argparse
import logging

import numpy as np
import tensorflow.compat.v1 as tf

from gleams import config
from gleams.embed import embedder, spectrum


# Fix logging hijacking by Tensorflow/abseil.
# FIXME: https://github.com/abseil/abseil-py/issues/99
# FIXME: https://github.com/tensorflow/tensorflow/issues/26691
try:
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    pass
# Disable Tensorflow v1 deprecation warnings.
tf.logging.set_verbosity(tf.logging.ERROR)


logger = logging.getLogger('gleams')


def _declare_args() -> argparse.Namespace:
    """
    Initialize the command-line arguments and provide sensible default values
    if possible.

    Returns
    -------
    argparse.Namespace
        The command-line argument namespace.
    """
    # IO arguments.
    parser = argparse.ArgumentParser(description='Train the GLEAMS embedder')
    parser.add_argument('--filename_base_train', required=True,
                        help='base filename for the training features; all '
                             'files matching {base}.npz (experimental spectra '
                             'features), {base}_sim_pos.npz (simulated '
                             'positive features), and {base}_sim_neg.npz '
                             '(simulated negative features) should exist')
    parser.add_argument('--filename_base_val',
                        help='base filename for the validation features; all '
                             'files matching {base}.npz (experimental spectra '
                             'features), {base}_sim_pos.npz (simulated '
                             'positive features), and {base}_sim_neg.npz '
                             '(simulated negative features) should exist')
    parser.add_argument('--filename_model', required=True,
                        help='output GLEAMS model filename')

    parser.add_argument('--debug', action='store_true',
                        help='enable detailed debug logging')

    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')

    args = _declare_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Build the embedder model.
    logger.info('Compile the GLEAMS siamese neural network')
    num_precursor_features = (config.num_bits_precursor_mz +
                              config.num_bits_precursor_mass +
                              config.precursor_charge_max)
    num_fragment_features = spectrum.get_num_bins(config.fragment_mz_min,
                                                  config.fragment_mz_max,
                                                  config.bin_size)
    num_ref_spectra_features = config.num_ref_spectra
    emb = embedder.Embedder(num_precursor_features, num_fragment_features,
                            num_ref_spectra_features, config.lr,
                            args.filename_model)
    emb.build_siamese_model()

    # Make sure all feature files exist.
    base_filenames = [args.filename_base_train]
    if args.filename_base_val is not None:
        base_filenames.append(args.filename_base_val)
    else:
        logger.debug('No validation will be performed because no validation '
                     'features have been provided')
    for base_filename in base_filenames:
        for filename in [f'{base_filename}_real_pos.npz',
                         f'{base_filename}_real_neg.npz',
                         f'{base_filename}_sim_pos.npz',
                         f'{base_filename}_sim_neg.npz']:
            if not os.path.isfile(filename):
                raise ValueError(f'Missing file {filename}')
    # Load the training and validation data.
    logger.info('Load all encoded spectra features')
    x_train, y_train = _load_features(args.filename_base_train,
                                      num_precursor_features,
                                      num_fragment_features)
    x_val, y_val = (_load_features(args.filename_base_val,
                                   num_precursor_features,
                                   num_fragment_features)
                    if args.filename_base_val is not None
                    else (None, None))

    # Train the embedder.
    num_pos = y_train.sum()
    num_neg = len(y_train) - num_pos
    logger.info('Train the GLEAMS siamese neural network using %d training '
                'samples (%d positive training samples, %d negative '
                'training samples)', len(y_train), num_pos, num_neg)
    emb.train(x_train, y_train, config.batch_size, config.num_epochs,
              x_val, y_val)

    logger.info('Save the trained GLEAMS siamese neural network')
    emb.save()

    logger.info('Training completed')


def _load_features(base_filename: str, num_precursor_features: int,
                   num_fragment_features: int):
    with np.load(f'{base_filename}_real_pos.npz') as f_real_pos,\
            np.load(f'{base_filename}_real_neg.npz') as f_real_neg,\
            np.load(f'{base_filename}_sim_pos.npz') as f_sim_pos,\
            np.load(f'{base_filename}_sim_neg.npz') as f_sim_neg:
        # Load all features.
        x1 = np.concatenate((f_real_pos['arr_0'], f_real_neg['arr_0'],
                             f_sim_pos['arr_0'], f_sim_neg['arr_0']))
        x2 = np.concatenate((f_real_pos['arr_1'], f_real_neg['arr_1'],
                             f_sim_pos['arr_1'], f_sim_neg['arr_1']))
        x1 = [x1[:, :num_precursor_features],
              x1[:, num_precursor_features:
                    num_precursor_features + num_fragment_features],
              x1[:, num_precursor_features + num_fragment_features:]]
        x2 = [x2[:, :num_precursor_features],
              x2[:, num_precursor_features:
                    num_precursor_features + num_fragment_features],
              x2[:, num_precursor_features + num_fragment_features:]]
        x = [*x1, *x2]
        y = np.hstack([np.ones(len(f_real_pos['arr_0'])),
                       np.zeros(len(f_real_neg['arr_0'])),
                       np.ones(len(f_sim_pos['arr_0'])),
                       np.zeros(len(f_sim_neg['arr_0']))])

        return x, y


if __name__ == '__main__':
    main()

