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
from gleams.embed import encoder, spectrum
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
    parser = argparse.ArgumentParser(description='Encode MS/MS spectra')
    parser.add_argument('spectra_filenames', nargs='+',
                        help='input spectrum files (in the mzML or mzXML '
                             'format, optionally compressed using gzip or xz)')
    parser.add_argument('--metadata',
                        help='comma-separated metadata file with spectrum '
                             'identifications (expected columns: filename, '
                             'scan, sequence, charge)')
    parser.add_argument('--spectrum_pairs', action='store_true',
                        help='generate spectrum to be used as training data '
                             'for the GLEAMS neural network')
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
    spectra = {}
    for file_i, spec_file in enumerate(args.spectra_filenames, 1):
        spec_file_base = os.path.basename(spec_file)
        logger.info('Process file %s [%d/%d]', spec_file_base, file_i,
                    len(args.spectra_filenames))
        if metadata_df is None:
            metadata_file = None
        elif spec_file_base in metadata_df.index:
            metadata_file = metadata_df.loc[spec_file_base]
        else:
            logger.warning('File %s not specified in the metadata, '
                           'skipping...', spec_file_base)
            continue
        for spec in tqdm.tqdm(ms_io.get_spectra(spec_file),
                              desc='Spectra read', leave=False,
                              unit='spectra'):
            if ((metadata_file is None or
                 spec.identifier in metadata_file.index)
                    and spectrum.preprocess(spec, config.fragment_mz_min,
                                            config.fragment_mz_max).is_valid):
                spectra[spec.identifier] = spec

        if not args.spectrum_pairs:
            # No pairs needed, just encode and export the spectra.
            if len(spectra) > 0:
                features = [enc.encode(spec) for spec in spectra.values()]
                np.savez_compressed(os.path.splitext(spec_file)[0] + '.npz',
                                    np.vstack(features))
            else:
                logger.warning('No spectra selected for encoding')
        else:
            # Generate training spectrum pairs.
            if metadata_file is None:
                logger.warning('Unable to generate spectrum pairs because no '
                               'metadata with spectrum identifications has '
                               'been provided')
            elif len(spectra) == 0:
                logger.warning('Unable to generate spectrum pairs because no '
                               'spectra were selected for encoding')
            else:
                logger.info('Generate encoded spectrum pairs')
                pair_generator = encoder.PairGenerator().set_spectra(
                    spectra, metadata_file)
                pair_options = [(True, True, 'real_pos'),
                                (True, False, 'real_neg'),
                                (False, True, 'sim_pos'),
                                (False, False, 'sim_neg')]
                for real, positive, pair_str in pair_options:
                    spectra1, spectra2 = zip(*pair_generator.generate_pairs(
                        real, positive))
                    features1 = [enc.encode(spec) for spec in spectra1]
                    features2 = [enc.encode(spec) for spec in spectra2]
                    np.savez_compressed(
                        f'{os.path.splitext(spec_file)[0]}_{pair_str}.npz',
                        np.vstack(features1), np.vstack(features2))

    logger.info('Encoding completed')


if __name__ == '__main__':
    main()
