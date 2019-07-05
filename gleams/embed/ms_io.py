import gzip
import logging
import lzma
import os
from typing import Iterator, Sequence

from spectrum_utils.spectrum import MsmsSpectrum

from gleams.embed import mzml_io
from gleams.embed import mzxml_io

logger = logging.getLogger('gleams')


def get_spectra(filename: str, scan_nrs: Sequence[int] = None)\
        -> Iterator[MsmsSpectrum]:
    """
    Get the MS/MS spectra from the given file, optionally filtering by scan
    number.

    Supported file formats are mzML, mzXML, and MS2. Files can optionally be
    GZIP or LZMA compressed.

    Parameters
    ----------
    filename : str
        The file name from which to read the spectra.
    scan_nrs : Sequence[int]
        Only read spectra with the given scan numbers. If `None`, no filtering
        on scan number is performed.

    Returns
    -------
    Iterator[MsmsSpectrum]
        An iterator over the requested spectra in the given file.
    """
    basename, ext = os.path.splitext(filename.lower())
    # Unzip if necessary.
    close = False
    if ext == '.gz':
        source = gzip.open(filename)
        close = True
        ext = os.path.splitext(basename)[1]
    elif ext == '.xz':
        source = lzma.open(filename)
        close = True
        ext = os.path.splitext(basename)[1]
    else:
        source = filename

    if ext == '.mzml':
        spectrum_io = mzml_io
    elif ext == '.mzxml':
        spectrum_io = mzxml_io
    else:
        raise ValueError(f'Unknown spectrum file type with extension "{ext}"')

    for spec in spectrum_io.get_spectra(
            source, sorted(set(scan_nrs)) if scan_nrs is not None else None):
        spec.is_processed = False
        yield spec

    if close:
        source.close()
