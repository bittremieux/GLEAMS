import logging
from typing import Dict, IO, Iterator, Sequence, Union

from lxml.etree import LxmlError
from pyteomics import mzxml
from spectrum_utils.spectrum import MsmsSpectrum


logger = logging.getLogger('gleams')


def get_spectra(source: Union[IO, str], scan_nrs: Sequence[int] = None)\
        -> Iterator[MsmsSpectrum]:
    """
    Get the MS/MS spectra from the given mzXML file, optionally filtering by
    scan number.

    Parameters
    ----------
    source : Union[IO, str]
        The mzXML source (file name or open file object) from which the spectra
        are read.
    scan_nrs : Sequence[int]
        Only read spectra with the given scan numbers. If `None`, no filtering
        on scan number is performed.

    Returns
    -------
    Iterator[MsmsSpectrum]
        An iterator over the requested spectra in the given file.
    """
    with mzxml.MzXML(source) as f_in:
        # Iterate over a subset of spectra filtered by scan number.
        if scan_nrs is not None:
            def spectrum_it():
                for scan_nr in scan_nrs:
                    yield f_in.get_by_id(str(scan_nr))
        # Or iterate over all MS/MS spectra.
        else:
            def spectrum_it():
                for spectrum_dict in f_in:
                    if int(spectrum_dict.get('msLevel', -1)) == 2:
                        yield spectrum_dict

        for spectrum in spectrum_it():
            try:
                yield _parse_spectrum(spectrum)
            except (ValueError, LxmlError) as e:
                logger.warning(f'Failed to read spectrum {spectrum["id"]}: %s',
                               e)


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

    Raises
    ------
    ValueError: The spectrum can't be parsed correctly:
        - Not an MS/MS spectrum.
        - Unknown precursor charge.
    """
    scan_nr = int(spectrum_dict['id'])

    if int(spectrum_dict.get('msLevel', -1)) != 2:
        raise ValueError(f'Unsupported MS level {spectrum_dict["msLevel"]}')

    mz_array = spectrum_dict['m/z array']
    intensity_array = spectrum_dict['intensity array']
    retention_time = spectrum_dict['retentionTime']

    precursor_mz = spectrum_dict['precursorMz'][0]['precursorMz']
    if 'precursorCharge' in spectrum_dict['precursorMz'][0]:
        precursor_charge = spectrum_dict['precursorMz'][0]['precursorCharge']
    else:
        raise ValueError('Unknown precursor charge')
    activation = spectrum_dict['precursorMz'][0]['activationMethod']

    spectrum = MsmsSpectrum(str(scan_nr), precursor_mz, precursor_charge,
                            mz_array, intensity_array, None, retention_time)
    spectrum.activation = activation

    return spectrum
