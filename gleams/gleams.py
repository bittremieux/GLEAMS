import logging
import os
import shutil
import tempfile
from typing import List

# Set environment variables before import of packages that might use them.
# Set number of threads for NumExpr.
os.environ['NUMEXPR_MAX_THREADS'] = str(max(os.cpu_count(), 64))
# Prevent Tensorflow CUDNN errors.
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import click
import pandas as pd

# Initialize logging.
from . import logger as glogger
glogger.init()
# Initialize all random seeds before importing any packages.
from . import rndm
rndm.set_seeds()

from . import __version__
from . import config
from .nn import nn


logger = logging.getLogger('gleams')


@click.command('embed')
@click.argument('peak_in', nargs=-1, required=True)
@click.argument('embed_out', required=True)
def embed(peak_in: List[str], embed_out: str):
    """
    Convert MS/MS spectra in PEAK_IN to 32-dimensional embeddings written to
    EMBED_OUT using the GLEAMS deep learning model.

    Supported formats for peak files in PEAK_IN are: mzML, mzXML, MGF.

    EMBED_OUT must have the ".npy" extension.
    A metadata file with the same base name as EMBED_OUT and the ".parquet"
    extension will be created as well.
    """
    if len(peak_in) == 0:
        raise click.BadParameter('No input peak files specified')
    if os.path.splitext(embed_out)[1].lower() != ".npy":
        raise click.BadParameter('EMBED_OUT should have the ".npy" extension')

    logger.info('GLEAMS version %s', str(__version__))

    # Create temporary working directory.
    temp_dir = tempfile.mkdtemp()
    metadata_filename = os.path.join(temp_dir, 'metadata.parquet')
    embed_dir = os.path.join(temp_dir, 'embed')
    os.mkdir(embed_dir)
    # Create a metadata file with the file names.
    metadata = pd.DataFrame({'filename': peak_in})
    metadata['dataset'] = os.path.splitext(os.path.basename(embed_out))[0]
    metadata.to_parquet(metadata_filename, index=False)
    # Embed the spectra.
    precursor_encoding = {'num_bits_mz': config.num_bits_precursor_mz,
                          'mz_min': config.precursor_mz_min,
                          'mz_max': config.precursor_mz_max,
                          'num_bits_mass': config.num_bits_precursor_mass,
                          'mass_min': config.precursor_mass_min,
                          'mass_max': config.precursor_mass_max,
                          'charge_max': config.precursor_charge_max}
    fragment_encoding = {'min_mz': config.fragment_mz_min,
                         'max_mz': config.fragment_mz_max,
                         'bin_size': config.bin_size}
    reference_encoding = {'filename': config.ref_spectra_filename,
                          'preprocessing': {
                              'mz_min': config.fragment_mz_min,
                              'mz_max': config.fragment_mz_max,
                              'min_peaks': config.min_peaks,
                              'min_mz_range': config.min_mz_range,
                              'remove_precursor_tolerance':
                                  config.remove_precursor_tolerance,
                              'min_intensity': config.min_intensity,
                              'max_peaks_used': config.max_peaks_used,
                              'scaling': config.scaling},
                          'fragment_mz_tol': config.fragment_mz_tol,
                          'num_ref_spectra': config.num_ref_spectra}
    embedder_config = {'num_precursor_features': config.num_precursor_features,
                       'num_fragment_features': config.num_fragment_features,
                       'num_ref_spectra_features': config.num_ref_spectra,
                       'lr': config.lr}
    nn.embed(metadata_filename, config.model_filename, embed_out, embed_dir,
             precursor_encoding, fragment_encoding, reference_encoding,
             embedder_config, config.batch_size, config.charges)

    # Clean up intermediate files.
    shutil.rmtree(temp_dir)
