import logging
from typing import Dict, IO, Iterator, Sequence, Union

from lxml.etree import LxmlError
from pyteomics import mzml
from spectrum_utils.spectrum import MsmsSpectrum


logger = logging.getLogger('gleams')


def get_spectra(source: Union[IO, str], scan_nrs: Sequence[int] = None)\
        -> Iterator[MsmsSpectrum]:
    """
    Get the MS/MS spectra from the given mzML file, optionally filtering by
    scan number.

    Parameters
    ----------
    source : Union[IO, str]
        The mzML source (file name or open file object) from which the spectra
        are read.
    scan_nrs : Sequence[int]
        Only read spectra with the given scan numbers. If `None`, no filtering
        on scan number is performed.

    Returns
    -------
    Iterator[MsmsSpectrum]
        An iterator over the requested spectra in the given file.
    """
    with mzml.MzML(source) as f_in:
        # Iterate over a subset of spectra filtered by scan number.
        if scan_nrs is not None:
            f_in.build_id_cache()
            
            def spectrum_it():
                for scan_nr in scan_nrs:
                    yield f_in.get_by_id(
                        f'controllerType=0 controllerNumber=1 scan={scan_nr}')
        # Or iterate over all MS/MS spectra.
        else:
            def spectrum_it():
                for spectrum_dict in f_in:
                    if int(spectrum_dict.get('ms level', -1)) == 2:
                        yield spectrum_dict

        try:
            for spectrum in spectrum_it():
                try:
                    yield _parse_spectrum(spectrum)
                except ValueError as e:
                    pass
                    # logger.warning(f'Failed to read spectrum %s: %s',
                    #                spectrum['id'], e)
        except LxmlError as e:
            logger.warning('Failed to read file %s: %s', source, e)


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
        - Unknown scan number.
        - Not an MS/MS spectrum.
        - Unknown precursor charge.
    """
    spectrum_id = spectrum_dict['id']
    if 'scan=' in spectrum_id:
        scan_nr = int(spectrum_id[spectrum_id.find('scan=') + len('scan='):])
    else:
        raise ValueError(f'Failed to parse scan number')

    if int(spectrum_dict.get('ms level', -1)) != 2:
        raise ValueError(f'Unsupported MS level {spectrum_dict["ms level"]}')

    mz_array = spectrum_dict['m/z array']
    intensity_array = spectrum_dict['intensity array']
    retention_time = spectrum_dict['scanList']['scan'][0]['scan start time']

    precursor = spectrum_dict['precursorList']['precursor'][0]
    precursor_ion = precursor['selectedIonList']['selectedIon'][0]
    precursor_mz = precursor_ion['selected ion m/z']
    if 'charge state' in precursor_ion:
        precursor_charge = int(precursor_ion['charge state'])
    elif 'possible charge state' in precursor_ion:
        precursor_charge = int(precursor_ion['possible charge state'])
    else:
        raise ValueError('Unknown precursor charge')

    spectrum = MsmsSpectrum(str(scan_nr), precursor_mz, precursor_charge,
                            mz_array, intensity_array, None, retention_time)

    # This method of figuring out the activation type is very brittle. Because
    # the keys in this dictionary aren't ordered, if there are multiple keys I
    # will simply assign the first one I pull out, which is basically at
    # random. Then again, there really shouldn't be multiple activation types
    # that aren't "collision energy". Then again again, I'm sure we'll
    # encounter them eventually.
    for activation in precursor.get('activation', [None]):
        if activation != 'collision energy':
            spectrum.activation = activation
            break

    return spectrum
