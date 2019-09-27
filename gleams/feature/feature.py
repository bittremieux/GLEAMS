import logging
import os

import numpy as np
import pandas as pd

from gleams import config
from gleams.feature import encoder, spectrum
from gleams.ms_io import ms_io


logger = logging.getLogger('gleams')


def peaks_to_features(dataset: str, filename: str, psms: pd.DataFrame,
                      enc: encoder.SpectrumEncoder) -> None:
    peak_filename = os.path.join(
        os.environ['GLEAMS_HOME'], 'data', 'peak', dataset, filename)
    feat_filename = os.path.join(
        os.environ['GLEAMS_HOME'], 'data', 'feature',
        dataset, f'{os.path.splitext(filename)[0]}.npz')
    logger.info('Generate features for peaks file %s', filename)
    features = [enc.encode(spec) for spec in ms_io.get_spectra(peak_filename)
                if spec.identifier in psms.index and
                spectrum.preprocess(spec, config.fragment_mz_min,
                                    config.fragment_mz_max).is_valid]
    np.savez_compressed(feat_filename, np.vstack(features))


def convert_massivekb_peaks_to_features(massivekb_task_id: str):
    metadata = pd.read_csv(os.path.join(
        os.environ['GLEAMS_HOME'], 'data', 'massivekb',
        f'metadata_{massivekb_task_id}.csv'),
        index_col=['dataset', 'filename', 'scan'])

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

    for dataset, filename in zip(metadata['dataset'], metadata['filename']):
        peaks_to_features(dataset, filename, metadata.loc[[dataset, filename]],
                          enc)
