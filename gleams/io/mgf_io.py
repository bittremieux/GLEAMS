import logging
from typing import Dict, IO, Iterator, Sequence, Union

from pyteomics import mgf
from spectrum_utils.spectrum import MsmsSpectrum


logger = logging.getLogger('gleams')


def get_spectra(source: Union[IO, str], scan_nrs: Sequence[int] = None)\
        -> Iterator[MsmsSpectrum]:
    """
    Get the MS/MS spectra from the given MGF file, optionally filtering by
    scan number.

    Parameters
    ----------
    source : Union[IO, str]
        The MGF source (file name or open file object) from which the spectra
        are read.
    scan_nrs : Sequence[int]
        Only read spectra with the given scan numbers. If `None`, no filtering
        on scan number is performed.

    Returns
    -------
    Iterator[MsmsSpectrum]
        An iterator over the requested spectra in the given file.
    """
    with mgf.MGF(source) as f_in:
        # Iterate over a subset of spectra filtered by scan number.
        if scan_nrs is not None:
            def spectrum_it():
                for scan_nr, spectrum_dict in enumerate(f_in):
                    if scan_nr in scan_nrs:
                        yield spectrum_dict
        # Or iterate over all MS/MS spectra.
        else:
            def spectrum_it():
                yield from f_in

        for spectrum in spectrum_it():
            try:
                yield _parse_spectrum(spectrum)
            except ValueError as e:
                logger.warning(f'Failed to read spectrum '
                               f'{spectrum["params"]["title"]}: %s', e)


def _parse_spectrum(spectrum_dict: Dict) -> MsmsSpectrum:
    """
    Parse the Pyteomics spectrum dict.

    Parameters
    ----------
    spectrum_dict : Dict
        The Pyteomics spectrum dict to be parsed.

    Returns
    -------
    MsmsSpectrum
        The parsed spectrum.
    """
    identifier = spectrum_dict['params']['title']

    mz_array = spectrum_dict['m/z array']
    intensity_array = spectrum_dict['intensity array']
    retention_time = float(spectrum_dict['params']['rtinseconds'])

    precursor_mz = float(spectrum_dict['params']['pepmass'][0])
    if 'charge' in spectrum_dict['params']:
        precursor_charge = int(spectrum_dict['params']['charge'][0])
    else:
        raise ValueError('Unknown precursor charge')

    spectrum = MsmsSpectrum(str(identifier), precursor_mz, precursor_charge,
                            mz_array, intensity_array, None, retention_time)

    return spectrum
