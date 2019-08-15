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
    parser.add_argument('filename_spectra_enc_exp',
                        help='encoded experimental spectra filename (needs to '
                             'have a .npz extension)')
    parser.add_argument('filename_spectra_enc_sim_pos',
                        help='encoded simulated positive spectra filename '
                             'corresponding to the encoded experimental '
                             'spectra (needs to have a .npz extension)')
    parser.add_argument('filename_spectra_enc_sim_neg',
                        help='encoded simulated negative spectra filename '
                             'corresponding to the encoded experimental '
                             'spectra (needs to have a .npz extension)')
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

    # Load the training and validation data.
    for filename in [args.filename_spectra_enc_exp,
                     args.filename_spectra_enc_sim_pos,
                     args.filename_spectra_enc_sim_neg]:
        if os.path.splitext(filename)[1] != '.npz':
            raise ValueError(f'Invalid extension for the encoded spectra '
                             f'{filename}; should be .npz')
    # TODO: Include validation data.
    with np.load(args.filename_spectra_enc_exp) as f_exp,\
            np.load(args.filename_spectra_enc_sim_pos) as f_sim_pos,\
            np.load(args.filename_spectra_enc_sim_neg) as f_sim_neg:
        logger.info('Load all encoded spectra features')
        # Load all features.
        x_exp = f_exp['arr_0']
        x_sim_pos, x_sim_neg = f_sim_pos['arr_0'], f_sim_neg['arr_0']
        # Combine features and labels.
        x_exp = np.tile(x_exp, (2, 1))
        x_exp = [x_exp[:, :num_precursor_features],
                 x_exp[:, num_precursor_features:
                          num_precursor_features + num_fragment_features],
                 x_exp[:, num_precursor_features + num_fragment_features:]]
        x_sim = np.concatenate((x_sim_pos, x_sim_neg))
        x_sim = [x_sim[:, :num_precursor_features],
                 x_sim[:, num_precursor_features:
                          num_precursor_features + num_fragment_features],
                 x_sim[:, num_precursor_features + num_fragment_features:]]
        x_train = [*x_exp, *x_sim]
        y_train = np.hstack([np.ones(len(x_sim_pos)),
                             np.zeros(len(x_sim_neg))])

        # Train the embedder.
        logger.info('Train the GLEAMS siamese neural network using %d training'
                    ' samples (%d positive training samples, %d negative '
                    'training samples)', len(y_train), len(x_sim_pos),
                    len(x_sim_neg))
        emb.train(x_train, y_train, config.batch_size, config.num_epochs)

        logger.info('Save the trained GLEAMS siamese neural network')
        emb.save()

    logger.info('Training completed')


if __name__ == '__main__':
    main()

