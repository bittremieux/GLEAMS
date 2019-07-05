import argparse
import logging
import os

import numpy as np
import tqdm

from gleams.embed import spectrum
from gleams.embed import encoder
from gleams.embed import ms_io
from gleams.embed import config

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
                        help='output spectrum features file (the .npy '
                             'extension will be appended to the file name if '
                             'it does not already have one)')
    parser.add_argument('--out_metadata', type=argparse.FileType('w'),
                        help='output spectrum metadata file')

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
    parser.add_argument('--no_normalize', action='store_false',
                        help='don\'t normalize the spectrum vectors to unit '
                             'length (default: spectrum vectors are '
                             'normalized)')

    # Reference spectra encoding.
    parser.add_argument('--ref_spectra', default=config.ref_spectra_filename,
                        help='reference spectra file name (default: '
                             '%(default)s)')
    parser.add_argument('--bin_size_dot',
                        default=config.bin_size_ref_spec, type=float,
                        help='bin size used to divide the m/z range for '
                             'spectrum dot product (default: %(default)s '
                             'm/z)')
    parser.add_argument('--max_num_ref_spectra',
                        default=config.max_num_ref_spectra, type=int,
                        help='maximum number of reference spectra used '
                             '(default: all spectra in the reference spectra '
                             'file)')

    # Other arguments.
    parser.add_argument('--max_q', default=config.max_q, type=float,
                        help='maximum q-value threshold to consider a '
                             'spectrum (default: %(default)s)')
    parser.add_argument('--max_num_spectra', default=None, type=int,
                        help='maximum number of spectra to encode (default: '
                             'all spectra in the specified file(s))')
    # TODO: PSM metadata is currently ignored.
    parser.add_argument('--psm_file', type=argparse.FileType('r'),
                        help='file with search results (for metadata '
                             'annotation)')

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
            args.fragment_mz_min, args.fragment_mz_max, args.bin_size,
            not args.no_normalize),
        encoder.ReferenceSpectraEncoder(
            args.ref_spectra, args.fragment_mz_min, args.fragment_mz_max,
            args.bin_size_dot, not args.no_normalize, args.max_num_ref_spectra)
    ])

    features = []
    spec_i = 0
    for file_i, spec_file in enumerate(args.spectra_filenames):
        logger.info('Processing file %s [%d/%d]', os.path.basename(spec_file),
                    file_i, len(args.spectra_filenames))
        for spec in tqdm.tqdm(ms_io.get_spectra(spec_file),
                              desc='Spectra encoded', leave=False,
                              unit='spectra'):
            # TODO: Filter on q-value from metadata.
            if spectrum.is_valid(spectrum.preprocess(
                    spec, args.fragment_mz_min, args.fragment_mz_max)):
                features.append(enc.encode(spec))
                spec_i += 1

            if (args.max_num_spectra is not None and
                    spec_i == args.max_num_spectra):
                logger.info('Stopping early after encoding %d spectra',
                            args.max_num_spectra)
                break

    logger.info('Save encoded spectra to %s', os.path.basename(args.out))
    np.save(args.out, np.vstack(features))
    # TODO: Collect and store metadata.


if __name__ == '__main__':
    main()
