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

    # Precursor encoding.
    parser.add_argument('--precursor_num_bits_mz',
                        default=config.num_bits_precursor_mz, type=int,
                        help='number of bits used to encode the precursor m/z '
                             '(default: %(default)s)')
    parser.add_argument('--precursor_mz_min',
                        default=config.precursor_mz_min, type=float,
                        help='minimum value between which to scale the '
                             'precursor m/z (default: %(default)s m/z)')
    parser.add_argument('--precursor_mz_max',
                        default=config.precursor_mz_max, type=float,
                        help='maximum value between which to scale the '
                             'precursor m/z (default: %(default)s m/z)')
    parser.add_argument('--precursor_num_bits_mass',
                        default=config.num_bits_precursor_mass, type=int,
                        help='number of bits used to encode the precursor '
                             'neutral mass (default: %(default)s)')
    parser.add_argument('--precursor_mass_min',
                        default=config.precursor_mass_min, type=float,
                        help='minimum value between which to scale the '
                             'precursor neutral mass (default: %(default)s '
                             'Da)')
    parser.add_argument('--precursor_mass_max',
                        default=config.precursor_mass_max, type=float,
                        help='maximum value between which to scale the '
                             'precursor neutral mass (default: %(default)s '
                             'Da)')
    parser.add_argument('--precursor_charge_max',
                        default=config.precursor_charge_max, type=int,
                        help='number of bits to use to encode the precursor '
                             'charge (default: %(default)s)')

    # Fragment encoding.
    parser.add_argument('--fragment_mz_min',
                        default=config.fragment_mz_min, type=float,
                        help='minimum of the m/z range used for spectrum '
                             'vectorization (default: %(default)s m/z)')
    parser.add_argument('--fragment_mz_max',
                        default=config.fragment_mz_max, type=float,
                        help='minimum of the m/z range used for spectrum '
                             'vectorization (default: %(default)s m/z)')
    parser.add_argument('--bin_size',
                        default=config.bin_size, type=float,
                        help='bin size used to divide the m/z range for '
                             'spectrum vectorization (default: %(default)s '
                             'm/z)')

    # Reference spectra encoding.
    parser.add_argument('--ref_spectra', default=config.ref_spectra_filename,
                        help='reference spectra file name (default: '
                             '%(default)s)')
    parser.add_argument('--fragment_mz_tol',
                        default=config.fragment_mz_tol, type=float,
                        help='fragment m/z tolerance used to compute the '
                             'spectrum dot product against reference spectra '
                             '(default: %(default)s m/z)')
    parser.add_argument('--max_num_ref_spectra',
                        default=config.max_num_ref_spectra, type=int,
                        help='maximum number of reference spectra used '
                             '(default: all spectra in the reference spectra '
                             'file)')

    # Other arguments.
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
    parser.add_argument('--ms2pip_model', default=config.ms2pip_model,
                        help='MS2PIP model used to predict theoretical spectra'
                             ' (default: %(default)s)')

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
            args.precursor_num_bits_mz, args.precursor_mz_min,
            args.precursor_mz_max, args.precursor_num_bits_mass,
            args.precursor_mass_min, args.precursor_mass_max,
            args.precursor_charge_max),
        encoder.FragmentEncoder(
            args.fragment_mz_min, args.fragment_mz_max, args.bin_size),
        encoder.ReferenceSpectraEncoder(
            args.ref_spectra, args.fragment_mz_min, args.fragment_mz_max,
            args.fragment_mz_tol, args.max_num_ref_spectra)
    ])

    if args.metadata is not None:
        logging.info('Read metadata file %s', os.path.basename(args.metadata))
        metadata_df = pd.read_csv(args.metadata)
        metadata_df.set_index(['filename', 'scan'], inplace=True)
    else:
        metadata_df = None

    # Encode experimental spectra.
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
                    and spectrum.preprocess(spec, args.fragment_mz_min,
                                            args.fragment_mz_max).is_valid):
                features.append(enc.encode(spec))
                if metadata_df is not None:
                    peptides.append(
                        metadata_df.loc[(spec_file_base, int(spec.identifier)),
                                        'sequence'])
                    charges.append(spec.precursor_mz)
                spec_i += 1

            if (args.max_num_spectra is not None and
                    spec_i == args.max_num_spectra):
                logger.info('Stopping early after encoding %d spectra',
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
                args.ms2pip_model)
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
            # TODO: Properly get negative training peptides: Shuffle peptides?
            features = [enc.encode(spec) for spec in tqdm.tqdm(
                spectrum_simulator.simulate(peptides, charges),
                desc='Spectra encoded', leave=False, unit='spectra')]
            filename = f'{os.path.splitext(args.out)[0]}_sim_neg.npz'
            logger.info('Save simulated negative training spectra to %s',
                        os.path.basename(filename))
            np.savez_compressed(filename, np.vstack(features))

    logger.info('Encoding completed')


if __name__ == '__main__':
    main()
