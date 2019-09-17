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
import os

import numpy as np
import pandas as pd
import tqdm

from gleams import config
from gleams.embed import spectrum
from gleams.embed import encoder
from gleams.embed import theoretical
from gleams.ms_io import ms_io


# Fix logging hijacking by Tensorflow/abseil.
# FIXME: https://github.com/abseil/abseil-py/issues/99
# FIXME: https://github.com/tensorflow/tensorflow/issues/26691
try:
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    pass


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
    parser = argparse.ArgumentParser(description='Encode MS/MS spectra')
    parser.add_argument('spectra_filenames', nargs='+',
                        help='input spectrum files (in the mzML or mzXML '
                             'format, optionally compressed using gzip or xz)')
    parser.add_argument('--out', required=True,
                        help='output spectrum features file (the .npz '
                             'extension will be appended to the file name if '
                             'it does not already have one)')

    parser.add_argument('--max_num_spectra', default=None, type=int,
                        help='maximum number of spectra to encode (default: '
                             'all spectra in the specified file(s))')
    parser.add_argument('--metadata',
                        help='comma-separated metadata file with spectrum '
                             'identifications (expected columns: filename, '
                             'scan, sequence, charge)')
    parser.add_argument('--simulate_training_spectra', action='store_true',
                        help='simulate theoretical spectra to be used as '
                             'training data for the GLEAMS neural network')
    parser.add_argument('--debug', action='store_true',
                        help='enable detailed debug logging')

    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')

    args = _declare_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

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

    if args.metadata is not None:
        logging.info('Read metadata file %s', os.path.basename(args.metadata))
        metadata_df = pd.read_csv(args.metadata).astype({'scan': str})
        metadata_df.set_index(['filename', 'scan'], inplace=True)
    else:
        metadata_df = None

    # Read the spectra from the file(s).
    features, peptides, charges = [], [], []
    spec_i = 0
    for file_i, spec_file in enumerate(args.spectra_filenames, 1):
        spec_file_base = os.path.basename(spec_file)
        logger.info('Process file %s [%d/%d]', spec_file_base, file_i,
                    len(args.spectra_filenames))
        if metadata_df is not None and spec_file_base not in metadata_df.index:
            logger.warning('File %s not specified in the metadata, '
                           'skipping...', spec_file_base)
            continue
        for spec in tqdm.tqdm(ms_io.get_spectra(spec_file),
                              desc='Spectra encoded', leave=False,
                              unit='spectra'):
            if ((metadata_df is None or
                 (spec_file_base, int(spec.identifier)) in metadata_df.index)
                    and spectrum.preprocess(spec, config.fragment_mz_min,
                                            config.fragment_mz_max).is_valid):
                features.append(enc.encode(spec))
                if metadata_df is not None:
                    peptides.append(
                        metadata_df.loc[(spec_file_base, int(spec.identifier)),
                                        'sequence'])
                    charges.append(spec.precursor_mz)
                spec_i += 1

        if (args.max_num_spectra is not None and
                spec_i >= args.max_num_spectra):
            logger.info('Stopping early after encoding at least %d spectra',
                        args.max_num_spectra)
            break

    if len(features) > 0:
        logger.info('Save encoded spectra to %s', os.path.basename(args.out))
        np.savez_compressed(args.out, np.vstack(features))
    else:
        logger.warning('No spectra selected for encoding')

    # Encode corresponding positive and negative training theoretical spectra.
    if args.simulate_training_spectra:
        if metadata_df is None:
            logger.warning('Unable to simulate theoretical spectra because no '
                           'metadata including spectrum identifications has '
                           'been provided')
        elif len(peptides) == 0:
            logger.warning('Unable to simulate theoretical spectra because no '
                           'spectra were selected for encoding')
        else:
            spectrum_simulator = theoretical.SpectrumSimulator(
                config.ms2pip_model)
            logger.info('Simulate positive training examples for the '
                        'experimental spectra')
            features = [enc.encode(spec) for spec in tqdm.tqdm(
                spectrum_simulator.simulate(peptides, charges),
                desc='Spectra encoded', leave=False, unit='spectra')]
            filename = f'{os.path.splitext(args.out)[0]}_sim_pos.npz'
            logger.info('Save simulated positive training spectra to %s',
                        os.path.basename(filename))
            np.savez_compressed(filename, np.vstack(features))
            logger.info('Simulate negative training examples for the '
                        'experimental spectra')
            # Shuffle to get decoy peptides.
            features = [enc.encode(spec) for spec in tqdm.tqdm(
                spectrum_simulator.simulate(peptides, charges, True),
                desc='Spectra encoded', leave=False, unit='spectra')]
            filename = f'{os.path.splitext(args.out)[0]}_sim_neg.npz'
            logger.info('Save simulated negative training spectra to %s',
                        os.path.basename(filename))
            np.savez_compressed(filename, np.vstack(features))

    logger.info('Encoding completed')


if __name__ == '__main__':
    main()
